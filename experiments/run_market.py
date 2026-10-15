"""Experiment 2: live-market verification (regime-copy + BTC forecasting)."""
import json
import sys
import numpy as np
sys.path.insert(0, ".")
from src.market import (fetch_btc_daily, regime_copy_dataset, forecast_windows,
                        chrono_split, minmax_fit_transform, naive_last_value_baseline)
from src.models import make_model


def small_forecast_model(arch, window, hidden=16, seed=0):
    """Reuse copy-task models for forecasting by mapping continuous window to symbols?
    Cleaner: train a tiny linear-recurrent regressor with same gating ideas.
    Here we benchmark properly: plain/LSTM3/GRU regressors implemented inline with numpy."""
    return {"arch": arch, "window": window, "hidden": hidden, "seed": seed}


def train_regressor(arch, Xtr, ytr, Xte, yte, hidden=16, epochs=60, lr=3e-3, seed=0):
    rng = np.random.RandomState(seed)
    W_in = rng.randn(1, hidden) * 0.1
    W_rec = rng.randn(hidden, hidden) * 0.1
    b = np.zeros(hidden)
    V = rng.randn(hidden, 1) * 0.1
    by = np.zeros(1)
    # gate params for lstm3/gru simplified scalar-vector version
    if arch in ("lstm3", "gru"):
        Wg = rng.randn(hidden, hidden) * 0.1
        bg = np.zeros(hidden) + (3.0 if arch == "lstm3" else 0.0)
    else:
        Wg, bg = None, None
    n = len(Xtr)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in idx[: min(n, 256)]:
            x = Xtr[i]
            # forward
            h = np.zeros(hidden)
            c = np.zeros(hidden)
            hs, cs = [], []
            for t in range(len(x)):
                xin = W_in * x[t]
                if arch == "plain":
                    h = np.tanh(xin.sum(0) * 0 + x[t] * W_in.sum(0) * 0 + h @ W_rec.T + x[t] * W_in.ravel() * 0 + (W_in * x[t]).sum(0) + b)
                elif arch == "lstm3":
                    f = 1 / (1 + np.exp(-(h @ Wg.T + bg)))
                    u = np.tanh((W_in * x[t]).sum(0) + h @ W_rec.T + b)
                    c = f * c + 0.5 * u
                    h = np.tanh(c) * 0.7
                elif arch == "gru":
                    z = 1 / (1 + np.exp(-(h @ Wg.T + bg)))
                    u = np.tanh((W_in * x[t]).sum(0) + h @ W_rec.T + b)
                    h = z * h + (1 - z) * u
                hs.append(h.copy()); cs.append(c.copy())
            pred = h @ V + by
            err = pred - ytr[i]
            # crude SGD (direction-correct, keeps dependency on gating)
            gV = h[:, None] * err
            gby = err
            V -= lr * gV
            by -= lr * gby
    # eval
    def predict(X):
        outs = []
        for i in range(len(X)):
            x = X[i]
            h = np.zeros(hidden); c = np.zeros(hidden)
            for t in range(len(x)):
                if arch == "plain":
                    h = np.tanh((W_in * x[t]).sum(0) + h @ W_rec.T + b)
                elif arch == "lstm3":
                    f = 1 / (1 + np.exp(-(h @ Wg.T + bg)))
                    u = np.tanh((W_in * x[t]).sum(0) + h @ W_rec.T + b)
                    c = f * c + 0.5 * u
                    h = np.tanh(c) * 0.7
                elif arch == "gru":
                    z = 1 / (1 + np.exp(-(h @ Wg.T + bg)))
                    u = np.tanh((W_in * x[t]).sum(0) + h @ W_rec.T + b)
                    h = z * h + (1 - z) * u
            outs.append(float((h @ V + by).item()))
        return np.array(outs)
    ptr = predict(Xtr)
    pte = predict(Xte)
    def metrics(p, y):
        return {"MAE": float(np.mean(np.abs(p - y))),
                "RMSE": float(np.sqrt(np.mean((p - y) ** 2))),
                "MAPE": float(np.mean(np.abs((p - y) / (y + 1e-9))) * 100)}
    return {"train": metrics(ptr, ytr), "test": metrics(pte, yte)}


def run():
    out = {}
    prices, src = fetch_btc_daily(days=365)
    out["btc_source"] = src
    out["btc_n"] = len(prices)
    out["btc_last"] = prices[-1]
    print(f"[market] BTC source={src} n={len(prices)} last={prices[-1]:.2f}", flush=True)
    # A) regime-copy on real returns
    from src.training import train_copy  # noqa
    regime = {}
    for nf in [10, 30, 50]:
        seq, tgt = regime_copy_dataset(prices, n_fill=nf, seed=0)
        # quick train/test split accuracy with a small LSTM3 (forget open) vs plain
        accs = {}
        for arch, kw in [("plain", {}), ("lstm3", {"forget_bias": 3.0}), ("gru", {})]:
            m = make_model(arch, hidden=32, seed=7, **kw)
            # train 300 rounds on market-regime rows (resampled each round from same price history)
            import numpy as np
            rng = np.random.RandomState(11)
            for rnd in range(300):
                s2, t2 = regime_copy_dataset(prices, n_fill=nf, seed=rng.randint(1 << 30))
                loss, grads, _ = m.loss_and_grad(s2[:, :64], t2[:64])
                m.apply_grad(grads)
            sT, tT = regime_copy_dataset(prices, n_fill=nf, seed=999)
            logits, _ = m.forward_cache(sT)
            acc = float(((logits.argmax(1) == tT)).mean())
            accs[f"{arch}{kw}"] = round(acc, 4)
        regime[f"fill={nf}"] = accs
        print(f"[regime-copy] fill={nf} {accs}", flush=True)
    out["regime_copy"] = regime
    # B) forecasting benchmark
    X, y = forecast_windows(prices, window=30)
    Xtr, ytr, Xte, yte = chrono_split(X, y, 0.8)
    Xtr_s, Xte_s, lo, span = minmax_fit_transform(Xtr, Xte)
    ytr_s = (ytr - lo) / span
    yte_s = (yte - lo) / span
    out["forecast_split"] = {"n_train": len(Xtr), "n_test": len(Xte), "window": 30}
    out["baseline_last_value_norm"] = naive_last_value_baseline(Xte_s, yte_s)
    for arch in ["plain", "lstm3", "gru"]:
        r = train_regressor(arch, Xtr_s, ytr_s, Xte_s, yte_s)
        out[f"forecast_{arch}"] = r
        print(f"[forecast] {arch} test={r['test']}", flush=True)
    with open("results/market.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/market.json")
    return out


if __name__ == "__main__":
    run()
