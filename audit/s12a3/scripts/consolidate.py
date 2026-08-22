"""S12A3 consolidation. HARD sentinel gates first (review finding: sentinels must be
enforced, not advisory), then per-arm table and pre-registered decision bands."""
import json, glob, sys, numpy as np

O = "/users/3171356m/agcl_audit_s0/s12a3/out/"
U = {json.load(open(f))["unit"]: json.load(open(f)) for f in glob.glob(O + "unit_*.json")}
assert len(U) == 37, f"need 37 units, have {len(U)}"

# GATE 1: plumbing sentinel — exact reproduction of the frozen S11 baseline value
# (0.7565 is the rounded display constant; the frozen float is in RAW_ord.json)
frozen = json.load(open("/users/3171356m/agcl_audit_s0/s11/out/RAW_ord.json"))
fauc = frozen["result"]["auc"] if "result" in frozen else frozen["auc"]
p = U["plumb"]["result"]
assert p["auc"] == fauc, f"HARNESS DRIFT: plumb {p['auc']!r} != frozen {fauc!r}"
# GATE 2: arm-A bitwise sentinel from extraction metadata — all seeds
for s in (0, 1, 2):
    m = json.load(open(O + f"extract_A_s{s}.json"))
    assert m["matches_s12a1_bitwise"] is True, f"A_s{s} not bitwise vs S12A1"
# GATE 3: arm A must reproduce S12A2 arm X per-seed
S12A2X = [0.6046, 0.6224, 0.6381]
for s in (0, 1, 2):
    a = U[f"A_s{s}_ord"]["result"]["auc"]
    assert abs(a - S12A2X[s]) < 5e-4, f"A_s{s}_ord {a} != S12A2 X {S12A2X[s]}"
print("SENTINELS: plumb 0.7565 delta 0.0000 | A bitwise vs S12A1 3/3 | A == S12A2-X 3/3\n")

print(f"{'arm':<26}{'s0':>8}{'s1':>8}{'s2':>8}{'mean':>8}   LOSO mean")
DESC = dict(A="baseline norm=T pbr=T e32", B="norm=F           e32",
            C="pbr=F            e32", D="norm=F pbr=F     e32",
            E="baseline         e64", F="baseline        e128")
R = {}
for arm in "ABCDEF":
    o = [U[f"{arm}_s{s}_ord"]["result"]["auc"] for s in (0, 1, 2)]
    l = [U[f"{arm}_s{s}_loso"]["result"]["auc"] for s in (0, 1, 2)]
    R[arm] = (o, l)
    print(f"{arm} {DESC[arm]:<24}" + "".join(f"{x:8.4f}" for x in o)
          + f"{np.mean(o):8.4f}   {np.mean(l):.4f}")
best = max("ABCDEF", key=lambda a: np.mean(R[a][0]))
bm = float(np.mean(R[best][0]))
band = ("ENCODER RETENTION SOLVED" if bm >= 0.68 else
        "COMPRESSION BOTTLENECK REMAINS" if 0.60 <= bm <= 0.63 else
        "PARTIAL RETENTION GAIN (pre-registered gap band)" if 0.63 < bm < 0.68 else
        "DEGRADED (<0.60)")
print(f"\nbest arm {best} mean3 {bm:.4f}  ->  PRE-REGISTERED BAND: {band}")
json.dump(dict(best_arm=best, best_mean=bm, band=band,
               table={a: dict(ord=R[a][0], ord_mean=float(np.mean(R[a][0])),
                              loso=R[a][1], loso_mean=float(np.mean(R[a][1])))
                      for a in R},
               sentinels=dict(plumb_exact_vs_frozen=True, A_bitwise=True, A_eq_S12A2X=True)),
          open(O + "CONSOLIDATED.json", "w"), indent=1)
