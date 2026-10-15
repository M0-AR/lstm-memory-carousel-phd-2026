"""Gradient-factor probe: measures per-step learning-signal decay/growth.

Method: build a PlainRNN with controlled recurrent scale, run a filler-only
tail of length T, inject a unit error at the end and backpropagate it,
recording ||delta_{t-1}|| / ||delta_t|| at each step. The median ratio is the
'factor' quoted in the video (~0.6 plain, ~1.18 4x-init, ~1.01 2x-init,
~1.13 CEC, ~0.64 forget@0.5).
"""
import numpy as np
from .models import PlainRNN, CECAdditive, LSTM3Gate


def probe_plain_factor(hidden=32, steps=50, seed=0, wscale=1.0):
    m = PlainRNN(hidden=hidden, seed=seed, wscale=wscale)
    rng = np.random.RandomState(seed + 5)
    B = 32
    seq = rng.randint(8, 16, size=(steps, B))
    logits, cache = m.forward_cache(seq)
    hs = cache["hs"]
    # unit signal at end, random direction per sample
    delta = rng.randn(B, hidden)
    delta /= (np.linalg.norm(delta, axis=1, keepdims=True) + 1e-12)
    ratios = []
    dh = delta
    W = m.params["W"]
    for t in reversed(range(steps)):
        n0 = np.linalg.norm(dh, axis=1).mean()
        dt = dh * (1.0 - hs[t + 1] ** 2)
        dh = dt @ W
        n1 = np.linalg.norm(dh, axis=1).mean()
        ratios.append(float(n1 / (n0 + 1e-12)))
    return float(np.median(ratios)), float(np.mean(ratios)), ratios


def probe_predicted_growth(factor, steps=50):
    return factor ** steps


def cell_size_growth(model, n_fill, seed=0):
    """Mean |c| after letter vs after n_fill fillers (pile-up diagnostic)."""
    from .tasks import sample_copy_batch
    rng = np.random.RandomState(seed)
    seq, _ = sample_copy_batch(n_fill, 8, rng)
    if not hasattr(model, "forward_cache"):
        return None
    logits, cache = model.forward_cache(seq)
    if "CS" not in cache and "cs" not in cache:
        return None
    CS = cache.get("CS", cache.get("cs"))
    c0 = np.abs(CS[1]).mean()
    c1 = np.abs(CS[-1]).mean()
    return float(c0), float(c1), float(c1 / (c0 + 1e-12))
