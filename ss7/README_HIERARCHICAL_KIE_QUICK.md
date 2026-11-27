# V4 short pilot

Budget: 15 minutes per Actions job, 11 minutes for the Python experiment.
One seed group, 8 worlds (3 train, 1 fusion, 1 reference, 1 validation, 2 test),
0.16 days/world, two episodes/class/world. 20 held-out episodes, four/class.
Small validation set, coarse threshold budgets, no definitive quality claims.
Retains HMAC KIE, four aggregate proxies and compromise. Adds causal per-node
background normalization from an explicitly attack-free initial 30-window
commissioning period; this is a simulation assumption, not proven in production.
Five class-specific experts conditioned on node type learned only from training,
HGB for compromise, RF for others, zone RF corroboration on independent fusion
world. Independent alert budgets, 16% each of five local channels and 20% zone.
Primary 20 false window alerts/network-day, sensitivity 100 and 500.
Same-world local ablation, old-feature local ablation, flat RF comparator,
per-class channel AUROC/AP. No new protocol transaction state, no unknown-attack
test, no operator calibration. Full validation deferred until a promising pilot.
