# V12 controlled simulator accounting pilot
Compare legacy vs generic outcome-conditioned transaction cohort accounting
on same seed 256101, paired topology, workload, scenario schedule and main
RNG streams. New class inherits old simulator; old implementation untouched.
Initiation always counted, closed cohorts complete template, failed cohorts
may emit a partial suffix. Generic alternating direction is an assumption,
not a protocol-correct MAP/TCAP/ISUP state machine. Each hop updates both
endpoints, message totals are exact without proportional rescaling; hidden
contribution tracked by flow object identity instead of path/count matching.
Existing adaptive reporting mask and signed KIE channel retained unchanged.
Offered full-template queue load is still approximate, no delayed states or
real traces. Two additional disjoint worlds per mode contain matched benign
service-volume effects (same distributions, no compromise). Validation hard
negative included in calibration; test hard negative scored separately.
Same V7 and V9 classifiers, fixed three-window rule and false-window budget.
No direct access to outcome truth, benign control tag, or attack metadata as
model features. No claims of intent identification. Four generation shards
per mode, two evaluation jobs, 10/9-minute caps. 14 worlds per mode; 20 attack
episodes per mode, paired, not 40 independent attack episodes.
