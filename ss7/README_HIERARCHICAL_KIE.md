# SS7 hierarchical KIE experiment

Separate experiment derived from canonical ss7/fix.py and the existing jSS7 KIE lab branch.
Four external attack names are aggregate traffic-effect proxies, not protocol exploits
or proof of intent; adaptive compromise retains canonical masking and software key custody.
Legitimate background and normal events remain in every independent world.

KIE is serialized, signed, verified, modeled with separate poll/response loss and delay.
Expired, invalid and accepted reports are distinct. Cross-node reconciliation uses decoded
reports only, never exact simulator loss. Link counters are offered counts, not delivered
packets. Application-layer bytes are counted, not measured SS7 wire overhead.

Local RF and benign-only isolation forest -> zone leave-one-out peer scores ->
network context -> supervised fusion, with no hard gate. Four independent simulation seeds
train local models, train fusion, calibrate threshold and test, respectively. Paired zero-effect
control is mandatory; not equivalent to full role-wise statistical null test.
Compare flat RF and reimplemented hard cascade with train-only scaling on identical worlds,
plus hierarchy without cross-node residuals. Threshold uses validation only (20 false
window alarms per network day). Three consecutive windows confirm an episode.
Results include per-class episode and window recall, FPR, latency, AUROC, AP, predictions,
scenario manifests, KIE overhead and null gate. Three seeds are preliminary; no claim of 90%
recall or live operator validation. Synthetic channel parameters are not calibrated.
Transport overhead defaults to zero; separately measured Osmocom value can be supplied.
Earlier article, canonical simulator and real-stack lab are untouched.

Run: python -m ss7.hierarchical_kie --seed 6101 --days 0.5 --repetitions 2 --out results
