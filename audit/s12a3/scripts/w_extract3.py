"""S12A3 extraction worker: one unit = (arm, seed_idx). Atomic, resumable, deterministic.
Saves nodes + flattened ROI features + hashes + config. Arm A cross-checked vs S12A1."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a3/scripts"); import s12a3_core as M
sys.path.insert(0, "/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A1
import s7_core as C7

UNITS = [(a, s) for a in "ABCDEF" for s in (0, 1, 2)]

def run(arm, sidx):
    tag = f"{arm}_s{sidx}"
    fp = M.emb_path(arm, sidx)
    if os.path.exists(fp): print("DONE(skip)", tag); return
    t0 = time.time(); cfg = M.ARMS[arm]; seed = M.BASE + sidx
    df, FC, Xid, y, ids = M.load_inputs()
    m = M.build_encoder_arm(93, seed, cfg)
    sdh = A1.sd_hash(m)
    N = M.extract_nodes(m, Xid, FC)
    assert N.shape == (954, 90, cfg["emb_dim"])
    # deterministic reload: rebuild same seed -> identical weights AND identical forward
    m2 = M.build_encoder_arm(93, seed, cfg)
    assert A1.sd_hash(m2) == sdh, "weight rebuild not deterministic"
    N2 = M.extract_nodes(m2, Xid[:64], FC[:64])
    assert np.array_equal(N[:64], N2), "forward not bitwise deterministic"
    flat = N.reshape(954, -1)
    extra = {}
    if arm == "A":  # sentinel: must equal S12A1 identity final_postnorm bitwise
        ref = np.load(f"/users/3171356m/agcl_audit_s0/s12a1/out/emb_id_s{sidx}.npz")["final_postnorm"]
        extra["matches_s12a1_bitwise"] = bool(np.array_equal(ref.astype(np.float32), N))
        print("A sentinel bitwise vs S12A1:", extra["matches_s12a1_bitwise"])
    tmp = fp + ".tmp.npz"
    np.savez_compressed(tmp[:-4], nodes=N, flat=flat, y=y,
                        cfg=json.dumps(cfg), seed=seed, sd_hash=sdh)
    z = np.load(tmp); assert z["nodes"].shape == N.shape and np.isfinite(z["flat"]).all()
    os.replace(tmp, fp)
    meta = dict(unit=tag, arm=arm, seed=seed, cfg=cfg, sd_hash=sdh,
                nodes_sha=M.arr_sha(N), flat_sha=M.arr_sha(flat),
                shape=list(N.shape), finite=True, deterministic_reload=True,
                wall_s=round(time.time() - t0, 1), **extra,
                provenance=C7.provenance({"unit": f"S12A3_extract_{tag}"}))
    json.dump(meta, open(f"{M.S12A3}out/extract_{tag}.json", "w"), indent=1, default=str)
    print("DONE", tag, meta["nodes_sha"][:12], f"{meta['wall_s']}s")

if __name__ == "__main__":
    torch.set_num_threads(1)                    # bitwise CPU reference (audit convention)
    a, s = UNITS[int(sys.argv[1])]
    run(a, s)
