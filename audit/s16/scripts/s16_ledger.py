"""S16 EXPECTED SCIENTIFIC LEDGER, generated directly from s16_grid + s16_data.

The ledger is the authority on what a COMPLETE wave means. A top-level scheduler
state of COMPLETED is NOT scientific completion: SLURM reports COMPLETED whenever the
process exits 0, which a unit does even when every one of its folds failed and was
recorded as status=FAILED. Completion is defined here, cell by cell."""
import sys, os, json, hashlib
sys.path.insert(0,"/users/3171356m/A-GCL/audit/s16/scripts")
import s16_grid as G, s16_data as DAT

EXPECTED_UNITS = 159
EXPECTED_FOLDS_PER_UNIT = 9
EXPECTED_CELLS = EXPECTED_UNITS * EXPECTED_FOLDS_PER_UNIT   # 1431

def fold_tags():
    """The 9 fold tags every unit must produce, taken from the frozen splits."""
    d,_,_ = DAT.load("signed", where="ledger")
    tags = ([t for t,_,_ in DAT.folds(d,"lab")][:3]
          + [t for t,_,_ in DAT.folds(d,"site")][:3]
          + [t for t,_,_ in DAT.folds(d,"loso")][:3])
    return tags

def all_units():
    out=[]
    for br,U in (("main",G.MAIN),("ctrl",G.CTRL),("abl",G.ABL)):
        for u in U: out.append((G.unit_id(u), br, u))
    return out

def expected_ledger():
    """-> (cells, units, tags). cells is the exact set of (unit_id, fold_tag)."""
    units = all_units(); tags = fold_tags()
    ids = [u[0] for u in units]
    if len(set(ids)) != len(ids):
        dup = [i for i in set(ids) if ids.count(i)>1]
        raise AssertionError(f"grid produces DUPLICATE unit ids: {dup[:5]}")
    cells = {(uid, t) for uid,_,_ in units for t in tags}
    return cells, units, tags

def assert_grid_shape():
    """Fails loudly if the grid drifts from the pre-registered shape."""
    cells, units, tags = expected_ledger()
    problems=[]
    if len(units) != EXPECTED_UNITS:
        problems.append(f"units {len(units)} != {EXPECTED_UNITS}")
    if len(tags) != EXPECTED_FOLDS_PER_UNIT:
        problems.append(f"folds/unit {len(tags)} != {EXPECTED_FOLDS_PER_UNIT} ({tags})")
    if len(cells) != EXPECTED_CELLS:
        problems.append(f"cells {len(cells)} != {EXPECTED_CELLS}")
    return problems, cells, units, tags

def ledger_hash():
    cells,_,_ = expected_ledger()
    return hashlib.sha256("|".join(sorted(f"{u}::{t}" for u,t in cells)
                                   ).encode()).hexdigest()[:16]

if __name__=="__main__":
    probs, cells, units, tags = assert_grid_shape()
    print(f"units          : {len(units)}  (expected {EXPECTED_UNITS})")
    print(f"folds per unit : {len(tags)}  {tags}")
    print(f"unique cells   : {len(cells)}  (expected {EXPECTED_CELLS})")
    print(f"ledger hash    : {ledger_hash()}")
    import collections
    print("units by branch:", dict(collections.Counter(b for _,b,_ in units)))
    print("SHAPE OK" if not probs else "SHAPE PROBLEMS: "+"; ".join(probs))
    sys.exit(1 if probs else 0)
