"""DEVIATION-01 (as corrected by AMENDMENT A2) selector: freeze the top-K configs
from Stage-B LAB+SITE only.

Rule A2, reusing aggBC's own health definitions verbatim:
  - a cell (protocol, fold, input) is ELIGIBLE for config c iff aggBC.run_valid,
    i.e. the OUTER refit passes health AND >= MIN_VALID_INNER (3) of the 5 inner
    trainings pass health;
  - c is ADMISSIBLE iff eligible in ALL lab+site cells, so every ranked config is
    scored on an identical set of cells and n is constant across the ranking;
  - s(c) = mean over those cells of aggBC.inner_score(cell, c), the mean inner AUC
    over health-valid inner runs. INNER FOLDS ONLY; outer AUC is never read;
  - rank by s(c) desc, ties by grid_B catalogue order; take K = 30.

Rule A1 (pool every valid inner run, ignore per-cell eligibility) is superseded:
it ranked configs on wildly different n (4 to 400) and was dominated by configs
that almost never train. See DEVIATION_01.md AMENDMENT A2.

Refuses to run unless LAB and SITE are BOTH complete for every (fold, input)
cell. Writes B/TOPK.json once and never overwrites it.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sweep_lib as L, mlp_lib as M
import aggBC as G

K = int(os.environ.get("SWEEP_SMOKE_K", "30"))          # production K is 30 (DEVIATION_01); smoke only
OUT = L.ROOT + "B/TOPK.json"
_ALL = [M.cfg_id(c) for c in M.grid_B()]
CFG_B = _ALL[:int(os.environ["SWEEP_SMOKE_GRID"])] if os.environ.get("SWEEP_SMOKE_GRID") else _ALL
RANK = {c: i for i, c in enumerate(CFG_B)}

MIN_CELLS_FRAC = 0.5      # A3(ii): padding requires eligibility in a simple majority of cells

def main():
    if os.path.exists(OUT):
        print(f"TOPK.json already exists - refusing to overwrite. {OUT}"); return 3
    nin = len(json.load(open(L.ROOT + "B/inputs.json")))
    per_cell = {c: [] for c in CFG_B}; elig = {c: 0 for c in CFG_B}; cells = 0
    winners = []                                            # A3(i): honest per-fold winners
    for p in ("lab", "site"):
        folds, rows = G.load_stage("B", p)
        exp = len(folds) * nin
        if len(rows) != exp:
            print(f"{p}: INCOMPLETE - {len(rows)}/{exp} cells carry a full grid. STOP."); return 4
        _, hb = G.nested_B(folds, rows)
        if hb["unresolved_folds"]:
            print(f"{p}: UNRESOLVED FOLDS {hb['unresolved_folds']} - cannot take honest winners. STOP."); return 6
        for w in hb["chosen"]: winners.append((p, w["fold"], w["cfg"], w["inner"]))
        for key, r in sorted(rows.items()):
            cells += 1
            for c in CFG_B:
                if c not in r["inner"] or not G.run_valid(r, c): continue
                elig[c] += 1
                per_cell[c].append(G.inner_score(r, c, r["inner_raw"][c]))
    # (i) winners, unconditionally, in catalogue order
    win_cfgs = sorted({w[2] for w in winners}, key=lambda c: RANK[c])
    # (ii) padding: majority-eligible, ranked by mean inner over eligible cells
    need = max(0, K - len(win_cfgs)); floor = int(np.ceil(MIN_CELLS_FRAC * cells))
    pool = [c for c in CFG_B if c not in set(win_cfgs) and elig[c] >= floor]
    pad = [t[2] for t in sorted(((float(np.mean(per_cell[c])), RANK[c], c) for c in pool), key=lambda t: (-t[0], t[1]))][:need]
    sel = win_cfgs + pad
    print(f"cells (lab+site) = {cells} | winners = {len(win_cfgs)} | padding pool (>= {floor} cells) = {len(pool)} | selected = {len(sel)}")
    if len(sel) < K:
        print(f"ONLY {len(sel)} CONFIGS AVAILABLE, NEED K={K}. STOP."); return 5
    sc = {c: (float(np.mean(per_cell[c])) if per_cell[c] else float("nan")) for c in sel}
    doc = dict(deviation="DEVIATION_01", amendment="A3", K=K,
               rule=("union of (i) every config selected by aggBC.nested_B for any LAB or SITE outer fold, "
                     "unconditionally, and (ii) padding to K by mean aggBC.inner_score over eligible cells "
                     "among configs eligible (aggBC.run_valid) in >= a simple majority of cells; "
                     "inner folds only; ties by grid_B catalogue order"),
               min_cells_for_padding=floor, selected_from_cells=cells, n_padding_pool=len(pool),
               n_configs_ranked=len(CFG_B), inputs_n=nin,
               configs=sel,
               winners=[dict(protocol=p_, fold=f_, cfg=c_, inner=i_) for p_, f_, c_, i_ in winners],
               n_winners=len(win_cfgs),
               scores=[dict(cfg=c, mean_inner=sc[c], catalogue_idx=RANK[c], eligible_cells=elig[c],
                            source=("winner" if c in set(win_cfgs) else "padding")) for c in sel])
    L.aj(doc, OUT)
    print(f"wrote TOPK.json: K={K} = {len(win_cfgs)} winners + {len(pad)} padding")
    for c in sel[:len(win_cfgs)]:
        print(f"   WINNER  {c:<42} inner={sc[c]:.4f}  cells={elig[c]}/{cells}")
    for c in pad[:6]:
        print(f"   pad     {c:<42} inner={sc[c]:.4f}  cells={elig[c]}/{cells}")
    return 0

if __name__ == "__main__": sys.exit(main())
