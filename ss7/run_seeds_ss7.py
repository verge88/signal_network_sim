"""
run_seeds_ss7.py — мультисид-прогоны и критерий Уилкоксона для SS7.

Использование:
    python run_seeds_ss7.py                       # сиды 42..51
    python run_seeds_ss7.py 42 43 44 45 46
    python run_seeds_ss7.py --regen               # + перегенерация датасета
    python run_seeds_ss7.py --with-ensemble --no-svm
"""
from ms_multiseed_core import ProtocolSpec, cli

SS7 = ProtocolSpec(
    name="ss7",
    train_module="ss7_training_v7",
    sim_module="ss7_simulator_v7",
    data_path="ss7_dataset_v7.csv",
    compromise_attack="slave_compromise",
    attack_types=["sms_intercept", "location_track", "signaling_dos",
                  "irsf", "slave_compromise"],
    # соответствует temporal_features из ss7_training_v7.main()
    temporal_features=[
        "var_obs_total_messages", "var_obs_entropy",
        "var_obs_inbound_outbound_ratio", "var_obs_map_dominance_ratio",
        "autocorr_obs_total_messages", "integrity_fail_rate",
        "obs_destinations_cv",
    ],
    base_out="ss7_seed_runs",
)

if __name__ == "__main__":
    cli(SS7)
