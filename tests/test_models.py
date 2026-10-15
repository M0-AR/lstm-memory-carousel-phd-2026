import sys
sys.path.insert(0, ".")
from src.models import make_model
from src.tasks import sample_copy_batch
from src.gradient import probe_plain_factor


def test_param_counts():
    assert make_model("plain", hidden=32, seed=0).n_params() == 1832
    assert make_model("lstm2", hidden=32, seed=0).n_params() == 4968
    assert make_model("lstm3", hidden=32, seed=0).n_params() == 6536
    assert make_model("gru", hidden=32, seed=0).n_params() > 4900


def test_forward_shapes():
    import numpy as np
    rng = np.random.RandomState(0)
    for arch in ["plain", "cec", "lstm2", "lstm3", "gru"]:
        m = make_model(arch, hidden=8, seed=0)
        seq, tgt = sample_copy_batch(5, 4, rng)
        logits, _ = m.forward_cache(seq)
        assert logits.shape == (4, 8)


def test_loss_decreases_plain_short():
    import numpy as np
    rng = np.random.RandomState(1)
    m = make_model("plain", hidden=16, seed=0, lr=3e-3)
    seq, tgt = sample_copy_batch(5, 64, rng)
    l0, g0, _ = m.loss_and_grad(seq, tgt)
    for _ in range(20):
        seq, tgt = sample_copy_batch(5, 64, rng)
        loss, grads, _ = m.loss_and_grad(seq, tgt)
        m.apply_grad(grads)
    seq, tgt = sample_copy_batch(5, 64, rng)
    l1, _, _ = m.loss_and_grad(seq, tgt)
    assert l1 < l0


def test_gradient_factor_ordering():
    med1, _, _ = probe_plain_factor(wscale=1.0, seed=0)
    med4, _, _ = probe_plain_factor(wscale=4.0, seed=0)
    assert med1 < 1.0
    assert med4 > med1
