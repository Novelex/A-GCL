#!/usr/bin/env python3
"""
Download rois_aal.1D (ABIDE-I PCP, C-PAC pipeline, nofilt_noglobal strategy)
for exactly the subjects we already have edge/label data for.

Why nofilt_noglobal specifically (see docs/plan.md, docs/COMPLETE_PLAN.md.pdf
Step 1.1): ALFF needs the unfiltered signal's full spectrum -- filt_noglobal's
bandpass filter would remove the low-frequency components ALFF measures in
the first place. rois_aal.1D is already the ROI-averaged time series --
exactly what A-GCL's node-feature computation (detrend -> FFT -> 3 bands)
expects as input.

Subject list: derived from data/raw/ASD_ADJ and data/raw/NC_ADJ filenames
(strip the _adj.mat suffix to get the ABIDE FILE_ID), NOT a fresh phenotypic
CSV filter -- this guarantees every downloaded rois_aal.1D lines up 1:1 with
a subject we already have a PCC edge matrix and ASD/NC label for, rather than
risking a different subject set than what's already on disk.

Usage:
    python scripts/download_alff_rois.py --output-dir data/ALFF_need
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


S3_ROOT = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
PIPELINE = "cpac"
STRATEGY = "nofilt_noglobal"
DERIVATIVE = "rois_aal"
EXTENSION = ".1D"


@dataclass(frozen=True)
class DownloadTask:
    file_id: str
    label: str
    url: str
    destination: Path


def derivative_url(file_id: str) -> str:
    return (
        f"{S3_ROOT}/Outputs/{PIPELINE}/{STRATEGY}/{DERIVATIVE}/"
        f"{file_id}_{DERIVATIVE}{EXTENSION}"
    )


def subject_ids_from_existing_data(raw_dir: Path) -> list[tuple[str, str]]:
    """Read FILE_IDs from data/raw/ASD_ADJ and data/raw/NC_ADJ -- the exact
    subjects we already have edge matrices and labels for."""
    pairs: list[tuple[str, str]] = []
    for folder, label in [("ASD_ADJ", "ASD"), ("NC_ADJ", "NC")]:
        folder_path = raw_dir / folder
        if not folder_path.is_dir():
            raise FileNotFoundError(f"{folder_path} does not exist")
        for fname in sorted(os.listdir(folder_path)):
            if not fname.endswith("_adj.mat"):
                continue
            file_id = fname[: -len("_adj.mat")]
            pairs.append((file_id, label))
    if not pairs:
        raise RuntimeError(f"No subjects found under {raw_dir}")
    return pairs


def download_streaming(url: str, destination: Path, *, retries: int, timeout: int,
                        overwrite: bool) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return "skipped", destination.stat().st_size

    part_path = destination.with_name(destination.name + ".part")
    if overwrite:
        destination.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    for attempt in range(1, retries + 1):
        resume_from = part_path.stat().st_size if part_path.exists() else 0
        headers = {"User-Agent": "A-GCL-ALFF-downloader/1.0"}
        if resume_from > 0:
            headers["Range"] = f"bytes={resume_from}-"

        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=timeout) as response:
                status_code = getattr(response, "status", 200)
                if resume_from > 0 and status_code != 206:
                    write_mode, resume_from = "wb", 0
                else:
                    write_mode = "ab" if resume_from > 0 else "wb"

                with part_path.open(write_mode) as output_file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output_file.write(chunk)

            if not part_path.exists() or part_path.stat().st_size == 0:
                raise OSError("Downloaded file is empty.")

            os.replace(part_path, destination)
            status = "resumed" if resume_from > 0 else "downloaded"
            return status, destination.stat().st_size

        except HTTPError as exc:
            if exc.code == 416:
                logging.warning("HTTP 416 for %s, removing stale partial file.", destination.name)
                part_path.unlink(missing_ok=True)
            elif exc.code in {403, 404}:
                raise RuntimeError(f"Remote file unavailable (HTTP {exc.code}): {url}") from exc
            else:
                logging.warning("Attempt %d/%d failed for %s: HTTP %s",
                                 attempt, retries, destination.name, exc.code)
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            logging.warning("Attempt %d/%d failed for %s: %s",
                             attempt, retries, destination.name, exc)

        if attempt < retries:
            time.sleep(min(2 ** (attempt - 1), 30))

    raise RuntimeError(f"Failed after {retries} attempts: {url}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download rois_aal.1D (cpac, nofilt_noglobal) for our existing subject set.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"),
                         help="Directory containing ASD_ADJ/ and NC_ADJ/ (default: data/raw)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ALFF_need"),
                         help="Where to write downloaded .1D files (default: data/ALFF_need)")
    parser.add_argument("--workers", type=int, default=4, help="Simultaneous downloads (default: 4)")
    parser.add_argument("--retries", type=int, default=5, help="Max attempts per file (default: 5)")
    parser.add_argument("--timeout", type=int, default=180, help="Network timeout, seconds (default: 180)")
    parser.add_argument("--overwrite", action="store_true", help="Redownload files that already exist")
    parser.add_argument("--max-subjects", type=int, default=None, help="Limit for a quick test run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                         datefmt="%H:%M:%S")

    subjects = subject_ids_from_existing_data(args.raw_dir)
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    logging.info("Pipeline: %s | Strategy: %s | Derivative: %s%s", PIPELINE, STRATEGY, DERIVATIVE, EXTENSION)
    logging.info("Subjects to download: %d (from %s)", len(subjects), args.raw_dir)

    output_dir = args.output_dir.resolve()
    dest_dir = output_dir / DERIVATIVE
    dest_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        DownloadTask(
            file_id=file_id,
            label=label,
            url=derivative_url(file_id),
            destination=dest_dir / f"{file_id}_{DERIVATIVE}{EXTENSION}",
        )
        for file_id, label in subjects
    ]

    manifest_rows: list[dict[str, str]] = []
    completed = failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {
            executor.submit(download_streaming, task.url, task.destination,
                             retries=args.retries, timeout=args.timeout,
                             overwrite=args.overwrite): task
            for task in tasks
        }
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1
            try:
                status, size_bytes = future.result()
                error = ""
                logging.info("[%d/%d] %s | %s | %.1f KiB", completed, len(tasks),
                             task.file_id, status, size_bytes / 1024)
            except Exception as exc:
                status, size_bytes, error = "failed", 0, str(exc)
                failed += 1
                logging.error("[%d/%d] %s | FAILED | %s", completed, len(tasks), task.file_id, exc)

            manifest_rows.append({
                "FILE_ID": task.file_id, "label": task.label, "pipeline": PIPELINE,
                "strategy": STRATEGY, "derivative": DERIVATIVE, "local_path": str(task.destination),
                "url": task.url, "status": status, "size_bytes": str(size_bytes), "error": error,
            })

    manifest_rows.sort(key=lambda r: r["FILE_ID"])
    manifest_path = output_dir / "download_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    successful = len(tasks) - failed
    logging.info("Download finished: %d successful, %d failed.", successful, failed)
    logging.info("Manifest: %s", manifest_path)
    logging.info("Output: %s", dest_dir)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
