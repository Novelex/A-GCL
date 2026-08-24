"""S16 C6 unit table. Index order STABLE — never reorder."""
SEEDS = [20260818, 20260819, 20260820]
E_LEVELS = ["signed", "abs", "pos_zero", "shift"]
KH = {"BNT": 32, "WGIN": 128, "EDGEMLP": 256}
ARCH = {"A1":"WGIN","A3":"WGIN","A4":"WGIN","A5":"BNT","A6":"BNT","A7":"EDGEMLP"}
# 21 configs: A1,A4,A5,A6 x 4 E = 16, plus A3 at signed, plus A7 x 4 E
# A7 = the edge MLP, architecturally identical to S12A5 arm C. It is an ORDINARY
# ARM: the bridge interpretation is WITHDRAWN because the S12A5 and S16 training
# recipes differ, so any C6-vs-C2 gap reflects training size AND recipe jointly.
CONFIGS = ([(a,e) for a in ("A1","A4","A5","A6") for e in E_LEVELS]
           + [("A3","signed")] + [("A7",e) for e in E_LEVELS])
CTRL_REF = [("A6","BNT"), ("A4","WGIN")]          # one reference per architecture
CONTROLS = ["C-RAND","C-PERM","C-SHUF","C-ROI"]
ALFF_ABL = ["raw","perband","joint"]              # z is the default, already in MAIN

def main_units():
    return [dict(branch="main", arm=a, E=e, arch=ARCH[a], kh=KH[ARCH[a]],
                 mode=m, seed_idx=s, alff_mode="z")
            for (a,e) in CONFIGS for m in ("plain","fused") for s in range(3)]
def ctrl_units():
    return [dict(branch="ctrl", arm=a, E="signed", arch=ar, kh=KH[ar], mode="plain",
                 control=c, arm_type="control", seed_idx=s, alff_mode="z")
            for c in CONTROLS for (a,ar) in CTRL_REF for s in range(3)]
def abl_units():
    return [dict(branch="abl", arm="A1", E="signed", arch="WGIN", kh=KH["WGIN"],
                 mode="plain", seed_idx=s, alff_mode=am)
            for am in ALFF_ABL for s in range(3)]
MAIN, CTRL, ABL = main_units(), ctrl_units(), abl_units()
# Step-4 split by SPEED so fast tasks cannot leave slots idle behind slow WGIN folds.
BNTU  = [u for u in MAIN if u["arch"] in ("BNT","EDGEMLP")]      # array A, fast
WGINU = [u for u in MAIN if u["arch"]=="WGIN"] + ABL             # array B, ~6-8 min/fold
CTRLU = CTRL                                                     # array C, absorbs spare
def unit_id(u):
    p=[u["branch"],u["arm"],u["E"],u["mode"],f"s{u['seed_idx']}"]
    if "control" in u: p.insert(2,u["control"])
    if u.get("alff_mode","z")!="z": p.insert(3,"alff-"+u["alff_mode"])
    return "_".join(p)
if __name__=="__main__":
    print(f"MAIN {len(MAIN)}  CTRL {len(CTRL)}  ABL {len(ABL)}  TOTAL {len(MAIN)+len(CTRL)+len(ABL)} units")
    print(f"folds/unit 9  -> fold-runs: main {len(MAIN)*9} ctrl {len(CTRL)*9} abl {len(ABL)*9}"
          f" = {(len(MAIN)+len(CTRL)+len(ABL))*9}")
    for u in (MAIN[0],MAIN[-1],CTRL[0],ABL[0]): print("  ",unit_id(u))
