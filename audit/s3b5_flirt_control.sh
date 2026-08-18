#!/bin/bash
set -uo pipefail
W=/mnt/scratch/users/3171356m/muhammad-GraSTIACL/data
ATL=$W/software/atlases/aal_mask_pad.nii.gz
SIF=$W/software/containers/fsl_6.0.7.4.sif
IDENT=$W/raw/_validation/ident.mat
OUT=/users/3171356m/agcl_audit_s0/flirt_ctrl
mkdir -p $OUT
for SPEC in "Olin_0050102:full_classical_chunk01"; do
  FID=${SPEC%%:*}; CH=${SPEC##*:}
  IN=$W/dparsf_work/$CH/Results/ALFF_FunImgD/mALFFMap_${FID}.nii
  echo "=== FLIRT $FID ==="
  apptainer exec --bind /mnt/scratch:/mnt/scratch --bind /users/3171356m:/users/3171356m $SIF \
    flirt -in $IN -ref $ATL -out $OUT/${FID}_classical_in_atlas.nii.gz \
          -applyxfm -init $IDENT -interp trilinear
  echo "rc=$?"
done
ls -la $OUT
echo "=== ident.mat contents ==="; cat $IDENT
