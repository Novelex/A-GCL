# S12A4 DATA GATE REPORT — PASS
954 subjects (455 ASD / 499 NC), 90 ROIs. Loaded via S11 manifest ONLY (sha-asserted).
Subject ID hash == frozen S3C cohort hash (aca3d945...d68c9); S12A1/S12A2 used the same
loader (identical by construction). Labels bitwise-equal manifest. Frozen S3C folds
(splits sha 28fed44d...8916), each subject in exactly one test fold.
FC: loaded .mat FC vs canonical S11 X_fc (upper triangle): max abs diff = 0.0 (bitwise).
Graph conversion: frozen S5 M1_B cache edge_weight reshape(90,90) vs FC: 0 mismatches
(>1e-6) across all 954; node x vs canonical M1_B: max abs 0.0.
Loaded paths: S11 manifest + X_fc.npz (sha-asserted), s3c/X_sources.npz, s5/M1_B/processed/
M1_B_v1.pt (sha 312266b2...f844 asserted), per-subject .mat under canonical parents (listed
in out/DATA_GATE.json). No 956-cache, no duplicate subjects, no alternate processed files.
