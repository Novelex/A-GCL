"""Runtime proof of the real A-GCL model contract for ALL 9 P/O/C x B/C/D paths."""
import sys, numpy as np, torch
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s7"); import s7_core as C
print("CURRENT call site : agcl_ABIDE.py:98-102 / agcl_ABIDE_queue.py:229-233 -> GInfoMinMax(TUEncoder(...), args.emb_dim)")
print("ORIGINAL call site: bed5441:A-GCL/adgcl_edge.py:50-53                  -> GInfoMinMax(TUEncoder(...), args.emb_dim)")
print("--emb_dim default = 32 in BOTH  => second positional argument = 32\n")
ok=True
for p in ["P","O","C"]:
    for b in C.BRANCHES:
        m=C.build_model(p,0)
        ph=[l for l in m.proj_head if hasattr(l,"in_features")]
        x,ei,ew,bt=C.batch_graphs([0,1,2],b)
        with torch.no_grad(): h,z,_=m.encode(bt,x,ei,None,ew)
        a=(ph[0].in_features==32 and ph[0].out_features==32 and
           ph[-1].in_features==32 and ph[-1].out_features==32 and
           h.shape[-1]==32 and z.shape[-1]==32)
        ok&=a
        print(f"  {p}_{b}: proj[0] {ph[0].in_features}->{ph[0].out_features}  "
              f"proj[-1] {ph[-1].in_features}->{ph[-1].out_features}  "
              f"h{tuple(h.shape)} z{tuple(z.shape)}  {'OK' if a else 'FAIL'}")
        assert h.shape[-1]==32, f"{p}_{b} h dim"; assert z.shape[-1]==32, f"{p}_{b} z dim"
print("\nproj_head:", C.build_model("P",0).proj_head)
assert ok; print("\nCONTRACT_ASSERT_PASS  (h=[N,32], z=[N,32] for all 9 paths)")
