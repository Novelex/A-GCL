#!/bin/bash
# SUPERSEDED C2 launcher. REFUSES TO RUN.
# Its estimator's calibration FAILED (random encoder +0.0231 vs band [-0.01,+0.01]).
# It must never overwrite C2_PROBE.md or C2_PRECISION.md, which are marked
# SUPERSEDED - DO NOT QUOTE. Canonical replacement: sb_c2_bounded.sh
echo "REFUSED: sb_c2.sh is superseded; use sb_c2_bounded.sh" >&2
exit 2
