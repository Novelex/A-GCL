"""S15 unit tables. Index order is STABLE — never reorder."""
SEEDS = [20260818, 20260819, 20260820]
LOSSES = ["L-BCE", "L-AUC"]
LRS = [3e-4, 1e-3]
WDS = [1e-3, 1e-2]
BNT_K = [8, 32]
WGIN_HID = [128, 256]          # amendment A1: {64,128} violated hidden >= D
# arm -> (arch, feature spec)
ARMS = {
    "W1": ("WGIN", "alff"), "W2": ("WGIN", "fcrow"), "W3": ("WGIN", "fcrow+alff"),
    "B1": ("BNT", "fcrow"),  "B2": ("BNT", "fcrow+alff"), "B3": ("BNT", "alff+onehot"),
}
def main_units():
    U = []
    for arm in ("B1", "B2", "B3"):
        for loss in LOSSES:
            for lr in LRS:
                for wd in WDS:
                    for k in BNT_K:
                        for s in range(3):
                            U.append(dict(branch="main", arm=arm, arch="BNT", loss=loss,
                                          lr=lr, wd=wd, K_or_hidden=k, seed_idx=s))
    for arm in ("W1", "W2", "W3"):
        for loss in LOSSES:
            for lr in LRS:
                for wd in WDS:
                    for hd in WGIN_HID:
                        for s in range(3):
                            U.append(dict(branch="main", arm=arm, arch="WGIN", loss=loss,
                                          lr=lr, wd=wd, K_or_hidden=hd, seed_idx=s))
    return U
MID = dict(loss="L-BCE", lr=3e-4, wd=1e-3)     # fixed mid config for controls
def ctrl_units():
    U = []
    for ctrl in ("C-PERM", "C-SHUF", "C-ROI", "C-RAND"):
        for arm, arch, kh in (("B2", "BNT", 32), ("W2", "WGIN", 128)):
            for s in range(3):
                U.append(dict(branch="ctrl", arm=arm, arch=arch, control=ctrl,
                              arm_type="control", K_or_hidden=kh, seed_idx=s, **MID))
    return U
def tran_units():
    U = []
    for mode in ("T1", "T2", "T3"):
        for arm, arch, kh in (("B2", "BNT", 32), ("W2", "WGIN", 128)):
            for s in range(3):
                U.append(dict(branch="tran", arm=arm, arch=arch, mode=mode,
                              K_or_hidden=kh, seed_idx=s, leakage=(mode == "T3"), **MID))
    return U
MAIN, CTRL, TRAN = main_units(), ctrl_units(), tran_units()
def unit_id(u):
    p = [u["branch"], u["arm"], u["arch"], f"kh{u['K_or_hidden']}", u["loss"],
         f"lr{u['lr']:g}", f"wd{u['wd']:g}", f"s{u['seed_idx']}"]
    if "control" in u: p.insert(2, u["control"])
    if "mode" in u: p.insert(2, u["mode"])
    return "_".join(p)
if __name__ == "__main__":
    print(f"MAIN {len(MAIN)}  CTRL {len(CTRL)}  TRAN {len(TRAN)}")
    print("main[0] :", unit_id(MAIN[0])); print("main[-1]:", unit_id(MAIN[-1]))
    print("ctrl[0] :", unit_id(CTRL[0])); print("tran[0] :", unit_id(TRAN[0]))
