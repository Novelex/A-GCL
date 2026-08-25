#!/bin/bash
# DEPRECATED legacy E2E launcher. REFUSES TO RUN.
# It invoked _e2e.py, which used the DEFAULT PRODUCTION NAMESPACE.
# Canonical launcher: sb_e2e_arr.sh (array) + sb_e2e_chk.sh (checker), S16_NS=e2e.
echo "REFUSED: sb_e2e.sh is deprecated; use sb_e2e_arr.sh + sb_e2e_chk.sh" >&2
exit 2
