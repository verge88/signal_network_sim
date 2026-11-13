"""Fixed-parameter V4 validation using the already selected V3 operating point."""
import argparse
import experiments.distributed_5g.temporal_graph_v4_development as screen

SEEDS=screen.SEEDS

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for seed in SEEDS:
        p.add_argument(f"--seed-{seed}",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()
    screen.ACCUM_MULTIPLIERS=(1.50,)
    screen.ALPHAS=(0.00015,)
    screen.DECAYS=(0.75,)
    screen.CLUSTER_ORDERS=(2,)
    roots={seed:getattr(a,f"seed_{seed}") for seed in SEEDS}
    screen.run(roots,a.output)
