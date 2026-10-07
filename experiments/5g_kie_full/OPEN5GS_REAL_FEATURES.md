# Real Open5GS SBI feature-bridge iteration

This iteration starts from the successful live Open5GS GitHub Actions run.

The goal is to move from infrastructure validation to real SBI feature extraction:
- stable baseline capture;
- repeated UDM restarts;
- repeated HTTP/2 bursts to live NRF;
- 2-second SBI feature windows;
- benign-only Isolation Forest training/calibration;
- phase-wise recall / false-alarm reporting;
- no synthetic filling of unavailable KIE/self-report fields.

This iteration intentionally treats controlled service restarts and NRF query bursts as
test perturbations, not as claims that those operations are themselves attacks.
