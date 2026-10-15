"""Experiment 1: synthetic copy-task benchmark (the video's core claim)."""
import json
import sys
import numpy as np
sys.path.insert(0, ".")
from src.models import make_model
from src.training import train_copy
from src.gradient import probe_plain_factor


def run(fast=True):
    out = {"gradient_factors": {}, "copy_results": {}}
    for wscale, tag in [(1.0, "1x"), (2.0, "2x"), (4.0, "4x")]:
        med, mean, _ = probe_plain_factor(wscale=wscale, seed=1)
        out["gradient_factors"][tag] = {"median": med, "mean": mean,
                                        "growth_50": med ** 50}
        print(f"[grad] plain wscale={tag} factor={med:.3f} growth50={med**50:.2e}", flush=True)
    if fast:
        configs = [
            ("plain", {}, [5, 30, 50], 300, 2),
            ("cec", {}, [50], 300, 2),
            ("lstm2", {"input_bias": -4.0}, [50, 200], 400, 2),
            ("lstm3", {"input_bias": -2.0, "forget_bias": 0.0}, [50], 400, 2),
            ("lstm3", {"input_bias": -2.0, "forget_bias": 3.0}, [50], 400, 2),
            ("gru", {}, [50], 600, 2),
        ]
    else:
        configs = [
            ("plain", {}, [5, 10, 24, 30, 50], 2000, 5),
            ("cec", {}, [50, 200], 2000, 5),
            ("lstm2", {"input_bias": 0.0}, [50, 200, 500, 1000], 2000, 5),
            ("lstm2", {"input_bias": -2.0}, [50, 200, 500, 1000], 2000, 5),
            ("lstm2", {"input_bias": -4.0}, [50, 200, 500, 1000], 2000, 5),
            ("lstm2", {"input_bias": -6.0}, [50, 200, 500, 1000], 2000, 5),
            ("lstm3", {"input_bias": -2.0, "forget_bias": 0.0}, [50, 200], 2000, 5),
            ("lstm3", {"input_bias": -2.0, "forget_bias": 3.0}, [50, 200, 500, 1000], 2000, 5),
            ("gru", {}, [50, 200, 500, 1000], 2000, 5),
        ]
    for arch, kw, fills, rounds, runs in configs:
        for nf in fills:
            solved = 0
            accs = []
            for r in range(runs):
                m = make_model(arch, hidden=32, seed=100 + r, **kw)
                res = train_copy(m, nf, max_rounds=rounds, log_every=rounds)
                accs.append(res["final_acc"])
                solved += int(res["solved"])
            key = f"{arch}{kw}|fill={nf}"
            out["copy_results"][key] = {"solved": f"{solved}/{runs}", "accs": accs}
            print(f"[{arch}{kw}] fill={nf} solved={solved}/{runs} accs={[round(a,3) for a in accs]}", flush=True)
    with open("results/synthetic.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/synthetic.json")
    return out


if __name__ == "__main__":
    run(fast=("--full" not in sys.argv))
