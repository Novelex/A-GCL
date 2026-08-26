"""RETIRED — superseded by test_final.py.

test_gate4 built a fixture that predates the strict bundle validator: its cells had no
prediction file, no checkpoint and no complete manifest, so under the current collector
every case is rejected as `bundle_invalid` BEFORE reaching the specific key it asserted.
That is precisely the masked-test failure mode recorded as D15 — a test rejected for the
wrong reason is not a passing test.

Its twelve cases are strictly subsumed by test_final.py, which uses a bundle-complete
159-unit fixture and asserts the SPECIFIC rejection reason for each:
    missing_unit / missing_fold          -> ledger comparison in H14 (1431-cell fixture)
    duplicate_fold / unexpected_cell     -> collector guards, exercised by H14
    failed_record                        -> collector guard, exercised by H14
    malformed_json                       -> H21 corrupted prediction, H22 corrupted result
    provenance absent / incompatible     -> H20 missing prediction, H25-H30 field mismatches
    poison_marker                        -> collector guard, exercised by H14
    tally disagreement / skipped         -> H18 missing TALLY, H19 missing UNIT.done
    wrong_namespace                      -> collector guard + H23 identity mismatch
This file intentionally exits NONZERO so it can never be reported as a silent pass.
Renamed out of the `test_*` namespace in Pass 4 (defect D56): a file that is expected
to fail must not sit in the regression suite, where it trains reviewers to ignore red.
"""
import sys
sys.stderr.write("RETIRED: test_gate4.py is superseded by test_final.py (see docstring).\n")
sys.exit(2)
