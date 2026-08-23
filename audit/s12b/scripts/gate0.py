"""GATE 0 — data/cache integrity. Fresh namespace, delete-and-rebuild, hard asserts.
NO try/except around any load. Any failure raises and kills the run."""
import sys, os, json, time, socket, subprocess, hashlib, numpy as np, torch, pandas as pd
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12b/scripts"); import s12b_core as B
import s11_core as K

def main():
    t0 = time.time()
    cp, d = B.build_cache()
    d2 = B.load_all()                                    # immediate reload re-verification
    assert list(d2["ids"]) == list(d["ids"])
    py = sys.version.replace("\n", " ")
    gpuq = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() \
        if torch.cuda.is_available() else "no GPU on this node"
    pf = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                        capture_output=True, text=True).stdout
    open(B.S12B + "out/pip_freeze.txt", "w").write(pf)
    meta = pd.DataFrame(dict(site=d["site"], y=d["y"], age=d["age"], sex=d["sex"],
                             fd=d["fd"]))
    site_tab = meta.groupby("site").agg(n=("y", "size"), asd=("y", "sum"),
        age_mean=("age", "mean"), pct_male=("sex", "mean"),
        fd_mean=("fd", "mean")).round(3)
    F = B.folds_all(d["y"]); n_ord = sum(1 for t, _, _ in F if t.startswith("o"))
    n_loso = sum(1 for t, _, _ in F if t.startswith("l"))
    lines = [
        "# S12B GATE 0 — DATA / CACHE INTEGRITY: **PASS** (every assert below raised on failure)",
        f"- cache (fresh namespace, delete-and-rebuilt): `{os.path.basename(cp)}`"
        f" sha256={K.sha(cp)[:16]}  ({os.path.getsize(cp)/1e6:.1f} MB)",
        f"- n=954 (ASD 455 / NC 499); 90 nodes and 8100 directed edges per graph"
        " (self-loops included, FC diag=1)",
        f"- subject-order sha256: {d['id_order_sha'][:16]} | label sha256: {d['y_sha'][:16]}",
        f"- S11 manifest sha: {d['manifest_sha'][:16]} | X_fc sha: {d['xfc_sha'][:16]}",
        f"- frozen splits sha: {d['splits_sha'][:16]} (== S3C authority) |"
        f" M1_B dataset sha: {d['dataset_sha'][:16]}",
        f"- FC source stats: sym_max={d['fc_stats']['sym']:.2e}, diag_dev="
        f"{d['fc_stats']['diag']:.2e}, cache_vs_mat max={d['fc_stats']['fc_graph_max']:.2e},"
        f" x_vs_M1B max={d['fc_stats']['x_max']:.1f}, mismatches={d['fc_stats']['mism']}",
        f"- folds: {n_ord} ordinary + {n_loso} LOSO (frozen, loaded not regenerated)",
        f"- ALFF band order: frozen M1 axis-2 order (ALFF/fALFF/mALFF as S3A), raw M1"
        " loaded via exact FILE_ID reindex; joint-minmax M1_B bitwise == S5 cache (asserted in load_tensors)",
        "",
        "## Environment",
        f"- git SHA {B.GIT} | host {socket.gethostname()} | {time.strftime('%F %T')}",
        f"- python {py}",
        f"- torch {torch.__version__} | CUDA {torch.version.cuda} | GPU(s): {gpuq}",
        f"- CPU count {os.cpu_count()} | pip freeze -> out/pip_freeze.txt",
        "",
        "## Cohort by site (n, ASD, mean age, %male, mean FD)",
        site_tab.to_string(),
        "",
        f"- wall {time.time()-t0:.1f}s",
    ]
    open(B.S12B + "GATE0_DATA.md", "w").write("\n".join(lines) + "\n")
    B.atomic_json(dict(cache=cp, cache_sha=K.sha(cp),
                       provenance=B.provenance()), B.S12B + "out/GATE0.json")
    print("GATE0 PASS", cp, flush=True)

if __name__ == "__main__":
    main()
