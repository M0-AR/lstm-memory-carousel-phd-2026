# REPRODUCIBILITY

- Python 3.10+ (tested 3.12), `pip install -r requirements.txt` (numpy/pandas/matplotlib/requests/pytest).
- Determinism: every RNG is `np.random.RandomState(seed)`; model seed + data seed separated
  (+1000/+777 offsets) so train/test rows never overlap. No shuffling across time (chrono splits).
- One command: `python experiments/run_all.py` (fast) or `--full` (paper-scale).
- Docker: `docker compose up` (fast) / `docker compose run full`. `results/` mounted to host.
- Offline: market module falls back to embedded BTC snapshot; synthetic needs no internet.
- Artifacts: `results/synthetic.json`, `results/market.json` (with data provenance fields).
- Tests: `python -m pytest tests/ -q` (4 asserts, <1s).
- Expected fast-runtime: synthetic ~3-8 min CPU, market ~2-5 min CPU (366-bar live pull).
- If CoinGecko rate-limits (429), code retries once then uses embedded snapshot and records
  `embedded-fallback` in `btc_source` — the run still completes and stays comparable.
