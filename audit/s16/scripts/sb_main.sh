#!/bin/bash
# DEPRECATED overlapping launcher. REFUSES TO RUN.
# The canonical production wave is exactly three arrays, split by speed:
#   sb_bnt.sh   (0-71)   BNT + EDGEMLP
#   sb_wgin.sh  (0-62)   WGIN + ALFF ablation
#   sb_ctrlu.sh (0-23)   controls
# Running sb_main.sh could DUPLICATE units already covered by those arrays.
echo "REFUSED: sb_main.sh is deprecated; use sb_bnt.sh / sb_wgin.sh / sb_ctrlu.sh" >&2
exit 2
