"""Live-market verification: no API keys, three free sources + embedded fallback.

Sources (tried in order):
  1. CoinGecko free /coins/bitcoin/market_chart (no key, rate-limited)
  2. Frankfurter ECB FX (no key) for EURUSD as second regime series
  3. Embedded snapshot (30 daily BTC closes 2026-09-15..2026-10-15, fetched 2026-10-15)
     so the benchmark is fully reproducible offline.

Two market tasks mirror the synthetic copy task:
  A) Regime-copy: marker = day-0 return quintile (8 bins ~ 8 letters);
     fillers = next N noisy market days; question = marker after N days.
     This is the direct real-data analogue of 'letter across N fillers'.
  B) Next-day forecasting: LSTM vs GRU vs Plain on BTC closes (window 30 -> next close),
     chronological 80/20 split, MinMax fit on train only (no leakage), MAE/RMSE/MAPE.
"""
import json
import urllib.request
import numpy as np

EMBEDDED_BTC = [
    75645.58, 75622.80, 76353.28, 81124.12, 81226.58, 81197.33, 86421.95, 86120.19,
    84550.14, 84249.39, 84069.47, 84260.05, 84171.14, 83514.23, 83583.21, 83648.98,
    84688.47, 84543.44, 84765.63, 86417.75, 85916.96, 85528.21, 83194.95, 81848.89,
    82563.29, 83019.84, 83574.57, 82134.49, 83458.77, 81578.94, 82198.68,
]
EMBEDDED_DATES = "2026-09-15..2026-10-15 (UTC, CoinGecko BTC/USD daily)"


def _get(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "lstm-memory-study/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def fetch_btc_daily(days=365):
    """Returns (prices list, source str). Tries CoinGecko, falls back to embedded."""
    try:
        raw = _get(f"https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily")
        j = json.loads(raw)
        prices = [p[1] for p in j["prices"]]
        return prices, f"coingecko-live(n={len(prices)})"
    except Exception as e:
        return list(EMBEDDED_BTC), f"embedded-fallback({e.__class__.__name__})"


def fetch_eurusd_series():
    try:
        raw = _get("https://api.frankfurter.app/2026-01-01..2026-10-15?from=EUR&to=USD")
        j = json.loads(raw)
        rates = [v["USD"] for k, v in sorted(j["rates"].items())]
        return rates, f"frankfurter-live(n={len(rates)})"
    except Exception as e:
        rng = np.random.RandomState(0)
        walk = 1.08 + np.cumsum(rng.randn(200) * 0.002)
        return list(walk), f"synthetic-fx-fallback({e.__class__.__name__})"


def regime_copy_dataset(prices, n_fill, n_bins=8, seed=0):
    """Build (seq, targets) analogue: marker bin from day-0 log-return quintile.
    Symbols: marker bin 0..7 as 'letter'; filler days quantised to 8..15 by return sign/mag.
    Returns seq [T,B], targets [B], plus metadata."""
    prices = np.asarray(prices, dtype=np.float64)
    rets = np.diff(np.log(prices))
    # need n_fill+1 returns per row: marker return + n_fill filler returns
    need = n_fill + 1
    if len(rets) < need + 5:
        # bootstrap longer series by resampling blocks (documented fallback for 31-day snapshot)
        rng0 = np.random.RandomState(seed)
        idx = rng0.randint(0, len(rets), size=need * 64)
        # handled per-row below
        pass
    rng = np.random.RandomState(seed)
    B = 256
    bins = np.quantile(rets, np.linspace(0, 1, n_bins + 1))
    seq = np.zeros((need, B), dtype=np.int64)
    tgt = np.zeros(B, dtype=np.int64)
    for b in range(B):
        start = rng.randint(0, max(1, len(rets) - need))
        window = rets[start:start + need]
        if len(window) < need:  # bootstrap wrap for short snapshot
            extra = rng.choice(rets, size=need - len(window))
            window = np.concatenate([window, extra])
        marker = int(np.clip(np.digitize(window[0], bins) - 1, 0, n_bins - 1))
        tgt[b] = marker
        seq[0, b] = marker
        for t in range(1, need):
            # filler symbol: quantise filler return into 8 bins -> 8..15
            f = int(np.clip(np.digitize(window[t], bins) - 1, 0, n_bins - 1))
            seq[t, b] = 8 + f
    return seq, tgt


def forecast_windows(prices, window=30):
    p = np.asarray(prices, dtype=np.float64)
    X, y = [], []
    for i in range(len(p) - window):
        X.append(p[i:i + window])
        y.append(p[i + window])
    return np.array(X), np.array(y)


def chrono_split(X, y, train_frac=0.8):
    n = len(X)
    k = int(n * train_frac)
    return X[:k], y[:k], X[k:], y[k:]


def minmax_fit_transform(tr, te):
    lo, hi = float(tr.min()), float(tr.max())
    span = (hi - lo) if hi > lo else 1.0
    return (tr - lo) / span, (te - lo) / span, lo, span


def naive_last_value_baseline(Xte, yte):
    pred = Xte[:, -1]
    return {"MAE": float(np.mean(np.abs(pred - yte))),
            "RMSE": float(np.sqrt(np.mean((pred - yte) ** 2))),
            "MAPE": float(np.mean(np.abs((pred - yte) / (yte + 1e-9))) * 100)}
