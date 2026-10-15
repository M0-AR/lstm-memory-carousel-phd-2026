"""Run every experiment end-to-end (synthetic fast + market live)."""
import sys
sys.path.insert(0, ".")
from experiments.run_synthetic import run as run_syn
from experiments.run_market import run as run_mkt

if __name__ == "__main__":
    full = "--full" in sys.argv
    print("== synthetic ==")
    run_syn(fast=not full)
    print("== market ==")
    run_mkt()
    print("ALL DONE. See results/synthetic.json + results/market.json")
