# Open5GS hierarchical calibration iteration

This iteration follows the first successful real Open5GS feature-bridge run.

## Why this iteration exists

The first real-data run proved that Open5GS SBI traffic can be captured and
converted into the same class of passive monitoring features used by the
synthetic 5G experiment. However, a 90-second baseline produced only 45
two-second windows and left too little independent calibration data for a
meaningful 1% false-positive target.

This iteration changes the experimental design rather than the detector family.

## Design

- real Open5GS control plane in GitHub Actions;
- 20-minute stable baseline (about 600 two-second windows);
- chronological four-way split:
  - 40% model training;
  - 20% channel conformal calibration;
  - 20% fusion conformal calibration;
  - 20% untouched benign holdout;
- global SBI Isolation Forest plus per-NF detectors;
- per-channel empirical conformal p-values;
- hierarchical fusion by the minimum channel p-value followed by an independent
  fusion-level conformal calibration;
- target nominal FPR = 1%;
- five UDM restart cycles and five NRF query-burst cycles;
- 30-second washout intervals are excluded from benign FPR;
- stable recovery is evaluated separately.

## Important interpretation

UDM restart and NRF GET bursts are controlled operational perturbations. They
are used to test whether real SBI changes are detectable; they are not asserted
to be 5G attacks.

The primary questions are:

1. Does a much larger real baseline make the 1% false-positive target
   empirically meaningful?
2. Does a global + per-NF hierarchy improve detection of distributed UDM
   lifecycle changes relative to a global-only detector?
3. Which NF channel (NRF, SCP, UDM, AMF, etc.) dominates each alarm?

KIE/self-report features remain unavailable in this iteration. No synthetic KIE
or self-report values are inserted into the real dataset.
