# V5 short calibration pilot

Two parallel 15-minute jobs, Python timeout 11 minutes. Same V4 world assignments
for 136101 (development/reused test group), fresh group 156101. Each uses 3 train,
1 fusion, 2 reference, 4 validation and 2 test worlds, 0.16 days/world, 2 episodes
per class/world. 20 test episodes/group. No independent confirmatory claim for
reused V4 test seeds. Fresh group still small and only preliminary.

Same signed KIE generator, features, five experts and zone RF as V4.
Compare V4 exact, V4 with added calibration, joint budget selection and
worst-validation-world threshold; flat RF comparator. Eight candidate combinations
of causal median/EWMA, partial lagged positive-drift correction, zone inclusion
are selected by validation recall under fixed 20 false window alerts/network-day.
No test scores computed before selection. Independent per-replication calibration.
The local-only candidate is an ablation, not a full hierarchy; if it wins,
report it as such. Also archive per-world test performance, sensitivity 100/500.
K=3 consecutive windows unchanged. No new protocol semantics or real-world
validation, no new unknown-attack claim. Fixed missing-only feature median in RF
without changing finite column imputation. V4 exact path remains same otherwise.
