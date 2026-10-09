# Iteration 4 execution corrections and provenance audit

**This is an append-only audit record, not an amendment to sampling thresholds,
masking protocol, physical packet witness, or frozen KIE response rules.**

1. **Workflow YAML substitution repair.** Initial GitHub Actions run
   [37936377350](https://github.com/verge88/signal_network_sim/actions/runs/37936377350)
   completed 29 scientific-contract tests, but download-artifact parameter names were
   emitted as literal `${seed}` and `${runid}`. Fixed 12 explicit archived
   artifact names and source-run IDs; no scientific code or input data changed.
2. **Original live-HMAC checkpoint versus additional archive rows.** Second run
   [37936503869](https://github.com/verge88/signal_network_sim/actions/runs/37936503869)
   completed the tests and downloaded all 12 archived Open5GS environments,
   but failed closed because archived report row count exceeded the contemporaneous
   verified receipt (first mismatch at seed 21011). No KIE outputs were reported
   from this failed execution. The subsequent correction preserves the
   original `verify_live_kie_phase_trial.py` receipt as the scientific
   provenance boundary: input the **first N report lines** where N was
   verified during the original run, require those N sample identities to
   pair exactly with the first N collector rows, and **exclude/count all rows
   appended after the checkpoint as unverified**. Report archived SHA256,
   N verified records, total archived records, and excluded tail separately
   by independent runner. Neither historical HMAC signatures nor reports
   appended after validation can be cryptographically re-verified using the
   destroyed ephemeral keys. Accordingly those unverified rows cannot
   enter the KIE scorer.
3. **Unmodified detector rule and intervention plan.** The physical
   candidate eligibility remains >=20 independent GET in sliding 2s,
   separated at gaps >4s, timestamp the first request. Actual original
   v1 and v2 score functions, reference/post windows and CPU tick threshold
   are unmodified. Scenario labels, masking and randomized timing are still
   join-only after scoring. Duplicate/missing/provenance issues fail closed.

The second change may slightly affect end-of-trial coverage if extra
sidecar samples occur after original validation. Such cases must be
classified unknown where no pre/post verified reports exist, **not**
counted as detector false negatives.

The original `OPEN5GS_NRF_PACKET_ONSET_KIE_PROTOCOL.md` is unchanged
to preserve preregistration history; this file is a transparent runtime
repair and is not evidence for a predetermined outcome.
