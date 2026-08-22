"""Phases F/G/H/I prep: all-954 Q1 proof, FC recovery+plumbing input, stage extraction
for old(3-feat) and identity(93-feat) x 3 seeds. eval+no_grad. Saves embeddings + hashes."""
import sys, os, json, numpy as np, torch, hashlib
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s12a1/scripts"); import s12a1_core as A
sys.path.insert(0,"/users/3171356m/agcl_audit_s0/s11"); import s11_core as K
import s7_core as C7
OUT=A.A1+"out"
z=np.load(OUT+"/gate_tensors.npz"); FC=z["FC"]; Xold=z["Xold"]; y=z["y"]
I90=np.eye(90,dtype=np.float32)
df,X_fc,y2,ids,gh=A.load_gate(); assert np.array_equal(y,y2)
SEEDS=[A.BASE,A.BASE+1,A.BASE+2]
sdh={}
def run_cond(model,Xn,tag):
    keep={k:[] for k in ("Q1","H1_preBN","H1_BN","H1_to_layer2","H2_preBN","H2_BN","final_postnorm","h","z")}
    for s0 in range(0,954,64):
        S=A.stages_for_subject_batch(model,Xn[s0:s0+64],FC[s0:s0+64])
        for k in keep: keep[k].append(S[k])
    E={k:np.concatenate(v) for k,v in keep.items()}
    for k,v in E.items(): assert np.isfinite(v).all(),(tag,k)
    return E
prov=C7.provenance({"unit":"S12A1_extract"})
# ---------- Phase F: Q1 proof over ALL 954 (seed-independent: Q1 has no parameters) ----------
m93=A.build_encoder(93,SEEDS[0])
Xid=np.concatenate([Xold,np.repeat(I90[None],954,0)],axis=2)
E0=run_cond(m93,Xid,"id_seed0")
Q=E0["Q1"][:,:,3:]
worst=0.0; worstrel=0.0; mism=0; dmin=9.0; dmax=-9.0
for i in range(954):
    ana=(I90+FC[i].T).astype(np.float32)
    d=np.abs(Q[i]-ana); worst=max(worst,float(d.max()))
    worstrel=max(worstrel,float((d/np.maximum(np.abs(ana),1e-12)).max()))
    if d.max()>1e-5: mism+=1
    dg=np.diag(Q[i]); dmin=min(dmin,float(dg.min())); dmax=max(dmax,float(dg.max()))
proof=dict(max_abs=worst,max_rel=worstrel,mismatching_subjects=mism,diag_min=dmin,diag_max=dmax,
    note_message_relu="message_relu=True is a layer-1 no-op: M1_B in [0,1] and I90 in {0,1} are "
    "non-negative, so relu(x_j)=x_j before edge weighting (S6-proven property, re-verified by the "
    "exactness of Q1 above).")
print(f"Q1 PROOF all-954: max_abs={worst:.2e} max_rel={worstrel:.2e} mism={mism} diag[{dmin},{dmax}]",flush=True)
assert worst<=1e-5 and mism==0
# ---------- Phase G: recover 4005 FC (transpose retained) ----------
REC=np.stack([ (Q[i]-I90).T[K.IU] for i in range(954) ])     # (Q-I) = FC^T -> transpose back, S11 pair order
assert REC.shape==(954,4005)
gmax=float(np.abs(REC-X_fc).max())
print(f"recovered FC vs canonical S11 X_fc: max_abs={gmax:.2e}",flush=True)
np.savez_compressed(OUT+"/plumbing_recovered_fc.npz",REC=REC.astype(np.float32),y=y)
# ---------- Phases H/I: all conditions x seeds ----------
for si,seed in enumerate(SEEDS):
    m93=A.build_encoder(93,seed); m3=A.build_encoder(3,seed)
    sdh[f"id_s{si}"]=A.sd_hash(m93); sdh[f"old_s{si}"]=A.sd_hash(m3)
    torch.save(m93.state_dict(),OUT+f"/enc93_s{si}.pt")       # donor MUST reuse this exact state
    for tag,m,Xn in (("id",m93,Xid),("old",m3,Xold)):
        E=run_cond(m,Xn,f"{tag}_s{si}")
        np.savez_compressed(OUT+f"/emb_{tag}_s{si}.npz",
            **{k:v.astype(np.float32) for k,v in E.items() if k!="Q1"},y=y)
        print(f"extracted {tag} seed{si}",flush=True)
json.dump(dict(provenance=prov,q1_proof=proof,recovered_fc_max_abs=gmax,
    state_dict_hashes=sdh),open(OUT+"/extract_meta.json","w"),indent=1,default=str)
open(OUT+"/EXTRACT_DONE","w").write("ok\n")
print("EXTRACT COMPLETE",flush=True)
