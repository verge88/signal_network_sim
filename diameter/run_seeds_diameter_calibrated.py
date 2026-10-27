"""Multi-seed training on the Open5GS-calibrated Diameter simulator.

The calibration changes S6a transaction sizes only. Traffic-rate priors,
command mix, attacks, topology and non-S6a sizing remain the canonical v1
defaults.
"""
import importlib

from ms_multiseed_core import ProtocolSpec, cli

_dt = importlib.import_module("diameter_training_calibrated")

OPEN5GS_CALIBRATION = {
    "s6a_transaction_size_bytes": 794.2608695652174,
    "s6a_command_transaction_size_bytes": {
        "AIR": 584.9230769230769,
        "ULR": 1066.4,
    },
}

CALIBRATION_PROVENANCE = {
    "method": "Open5GS EPC MME/HSS S6a software-in-the-loop calibration",
    "open5gs_commit": "f8aa307cdac2f1fd53e32fa96a1730d93bf8dc6d",
    "source_run_id": 40288325482,
    "source_artifact_id": 12634439202,
    "source_artifact_sha256":
        "f005a7b64b4ecf811706403fae20226948bdb6594d94010bb0146c571cc30244",
    "complete_transactions": 78,
    "fit_transactions": 46,
    "holdout_transactions": 32,
    "holdout_mae_bytes": 6.684615384615341,
    "holdout_median_relative_error": 0.010041784372528633,
    "holdout_p90_relative_error": 0.022593,
    "counter_semantics": (
        "The simulator's legacy *_count/total_messages names are treated as "
        "modeled Diameter command operations/transactions: scenario procedure "
        "composition counts AIR/ULR/CCR-I operations, while an Open5GS S6a "
        "calibration target contains the paired request+answer Diameter PDUs. "
        "This is a simulator abstraction, not an operator-counter validation."
    ),
    "claim_boundary": (
        "Protocol-real Open5GS S6a size calibration; not operator traffic "
        "and not a calibration of traffic or attack priors."
    ),
}

DIAMETER_CALIBRATED = ProtocolSpec(
    name="diameter_open5gs_calibrated",
    train_module="diameter_training_calibrated",
    sim_module="diameter_sim_v1",
    data_path="diameter_dataset_v1.csv",
    compromise_attack="node_compromise",
    attack_types=[
        "location_tracking",
        "sms_data_interception",
        "signaling_dos",
        "fraud_profile",
        "node_compromise",
    ],
    temporal_features=list(_dt.DataPipeline.TEMPORAL_FEATURES),
    base_out="diameter_calibrated_seed_runs",
    sim_config_overrides=OPEN5GS_CALIBRATION,
    calibration_provenance=CALIBRATION_PROVENANCE,
)

if __name__ == "__main__":
    cli(DIAMETER_CALIBRATED)
