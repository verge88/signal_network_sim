# V8 synthetic event observer pilot
Two parallel 15-minute jobs, Python cap 11 minutes. Reused V7 group 196101 and
fresh group 216101; 12 worlds/group, 20 test episodes/group. Paired fixed V7
comparator: 70 trees, leaf 20, median, no drift adaptation, local-priority.
Optional measurement hook in canonical simulator does not change legacy random
streams or scenarios. Explicit independently trusted synthetic observer with
separate HMAC key and loss/latency model. Sampled generated transaction cohorts,
simulator aggregate success outcome used to emit delayed synthetic completion
events; failures expire after three windows. Capture 98%, assumed 80/15/5% delay,
not measured. Eight services, external/internal counterpart, starts/completions/
expiry/open counts. No transaction IDs, decoded packets, authorization or real
protocol state. No attack metadata or hidden counter passed to model. Independent
trusted observation is a new strong assumption; improvement cannot be attributed
to original KIE alone. Observer payload overhead separately recorded. Full
hierarchy, local/graph ablations, 20 false window alerts/network-day, three-window
confirmation, sensitivity 100/500. Mandatory null gate. No real transport,
operator calibration, or unknown attack claims.
