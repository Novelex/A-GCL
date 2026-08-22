# S12A1 DATA GATE — ALL PASS (blocking; executed before any probe)
git HEAD 1ece8166421fe83c97c0500573216704be0a88fc (audit/r0 local commits; production surface
unchanged vs 8cac2358 lineage) | authoritative order: frozen S11 manifest (sha 4f01b0ab...)
| check | result |
|---|---|
| N=954, ASD=455, NC=499, ROI=90 | PASS |
| duplicates / missing / extras | 0 / 0 / 0 |
| FILE_ID + label == manifest, all 954 | PASS (asserted per subject) |
| graph x [90,3], finite, == canonical M1_B | PASS, max_abs 0.00e+00 |
| FC [90,90], finite, symmetric, diag==1 (S4 tol) | PASS (2.2e-16 / 2.2e-16) |
| FC source == reconstructed graph edge_weight, all 954 | PASS, max_abs 2.98e-08, 0 mismatch |
| S11 hashes (manifest/X_fc/pair-map/splits/dataset/ROI) | PASS (asserted, S11 values) |
| historical caches (956 / data_dense_v3 / OLD / M2 / norm_matrix) | NOT LOADED — impossible by construction (exact-ID paths only) |
| unique source parent dirs | data/raw/ASD_ADJ, data/raw/NC_ADJ only |
