# V8 sharded reproducibility rerun

The previous fresh group 216101 timed out during generation, before training.
Reuse exactly the V8 generator, seeds, scenario parameters, observation, models,
feature sets, calibration and scoring. Four independent generation shards,
three worlds per shard, zero-effect gate on shard 0; each 9-minute job with
7-minute program cap. Evaluation only after all shards pass, 8-minute job,
6-minute program cap. Runner queue and transfer time can exceed overall 15
minutes, even though jobs are short. Generated world files are trusted binary
artifacts from the same workflow, with SHA-256 integrity and seed/role validation.
No change to observation sampling or random streams. Per-channel AUROC/AP and
availability on validation/test added for diagnostics only, no tuning.
No new protocol semantics or real-world validation; same synthetic trusted
observer assumption. No optimization of model or simulator algorithms claimed.
