"""
run_seeds_diameter.py — мультисид-прогоны и критерий Уилкоксона для Diameter.

Использование:
    python run_seeds_diameter.py                  # сиды 42..51
    python run_seeds_diameter.py 42 43 44 45 46
    python run_seeds_diameter.py --regen
"""
import importlib
from ms_multiseed_core import ProtocolSpec, cli

_dt = importlib.import_module("diameter_training_v1")

DIAMETER = ProtocolSpec(
    name="diameter",
    train_module="diameter_training_v1",
    sim_module="diameter_sim_v1",
    data_path="diameter_dataset_v1.csv",
    compromise_attack="node_compromise",
    attack_types=["location_tracking", "sms_data_interception",
                  "signaling_dos", "fraud_profile", "node_compromise"],
    temporal_features=list(_dt.DataPipeline.TEMPORAL_FEATURES),
    base_out="diameter_seed_runs",
)

if __name__ == "__main__":
    cli(DIAMETER)
