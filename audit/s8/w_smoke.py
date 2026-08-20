"""S8 smoke: 8 subjects, 2 seeds, 3 configs. Tests every training-path component."""
import sys, os, json, time, copy, numpy as np, torch
torch.set_num_threads(1)   # determinism: pin threads (SLURM jobs set OMP_NUM_THREADS=1)
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s8"); import s8_core as S
import s7_core as C7
OUT=S.S8+"smoke"; os.makedirs(OUT,exist_ok=True)
t0=time.time(); dl=S.load_dataset(range(8))                      # graph loading
rep={"provenance":C7.provenance({"unit":"S8_smoke"}),"configs":{}}
for cfg in ["P","O","C"]:
    for seed in [0,1]:
        m_init,v_init,_,_,_,_=S.build(cfg,seed)               # same seed -> same init
        p0=[p.detach().clone() for p in m_init.parameters()]
        v0=[p.detach().clone() for p in v_init.parameters()]
        model,view,curves=S.train(cfg,seed,epochs=2,dl=dl,bs=8)   # bs=8: drop_last would yield 0 batches from 8 graphs at bs=32       # fwd/VL/aug/loss/bwd/step x2 epochs
        dm=max(float((a-b).abs().max()) for a,b in zip(p0,[p.detach() for p in model.parameters()]))
        dv=max(float((a-b).abs().max()) for a,b in zip(v0,[p.detach() for p in view.parameters()]))
        fin=all(np.isfinite([v for k,v in ep.items() if k not in ("epoch","keep_mu")]).all()
                for ep in curves)                                 # keep_mu is nan BY DESIGN for O
        rep["configs"][f"{cfg}_s{seed}"]=dict(
            losses_finite=bool(fin), model_params_changed=dm>0, view_params_changed=dv>0,
            max_param_delta_model=dm, max_param_delta_view=dv,
            final=curves[-1])
        print(f"{cfg} s{seed}: finite={fin} dModel={dm:.2e} dView={dv:.2e} "
              f"view_loss={curves[-1]['view_loss']:.4f} model_loss={curves[-1]['model_loss']:.4f} "
              f"keep={curves[-1]['keep_sampled']:.4f}",flush=True)
# deterministic CPU check: identical seed -> bitwise identical embeddings
mA,_,_=S.train("C",0,epochs=2,dl=dl,bs=8); hA,zA,_=S.extract(mA,dl)
mB,_,_=S.train("C",0,epochs=2,dl=dl,bs=8); hB,zB,_=S.extract(mB,dl)
det=bool(np.array_equal(hA,hB) and np.array_equal(zA,zB))
rep["deterministic_cpu_rerun"]=det
print("deterministic CPU rerun (bitwise):",det,flush=True)
assert det, "SMOKE FAIL: nondeterministic"
rep["runtime_s"]=round(time.time()-t0,1)
C7.write_unit(OUT,"S8_SMOKE",payload_json=rep)                   # TEMP->validate->rename->DONE
json.load(open(OUT+"/S8_SMOKE.json"))                            # reopen
open(OUT+"/SMOKE_PASS","w").write(C7.git_head()+"\n")
print(f"S8 SMOKE PASS in {rep['runtime_s']}s",flush=True)
