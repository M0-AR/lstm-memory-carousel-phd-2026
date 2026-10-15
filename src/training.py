"""Training loops with the video's protocol: 64-row rounds, solved = >=99% on 1000 new rows."""
import numpy as np
from .tasks import sample_copy_batch, eval_accuracy, sample_stream_batch


def train_copy(model, n_fill, max_rounds=2000, batch=64, seed=0, log_every=200,
               eval_rows=1000, solve_thr=0.99, clip=5.0):
    rng = np.random.RandomState(seed + 1000)
    hist = []
    solved_at = None
    for rnd in range(1, max_rounds + 1):
        seq, tgt = sample_copy_batch(n_fill, batch, rng)
        loss, grads, _ = model.loss_and_grad(seq, tgt)
        model.apply_grad(grads, clip=clip)
        if rnd % log_every == 0 or rnd == 1:
            acc = eval_accuracy(model, n_fill, n_rows=eval_rows, seed=10000 + rnd)
            hist.append((rnd, float(loss), float(acc)))
            if acc >= solve_thr:
                solved_at = rnd
                break
    acc = eval_accuracy(model, n_fill, n_rows=eval_rows, seed=99999)
    if acc >= solve_thr and solved_at is None:
        solved_at = max_rounds
    return {"solved": acc >= solve_thr, "solved_at": solved_at,
            "final_acc": float(acc), "hist": hist}


def train_stream(model, length=300, max_rounds=1500, batch=32, seed=0, log_every=300, clip=5.0):
    rng = np.random.RandomState(seed + 777)
    for rnd in range(1, max_rounds + 1):
        seq, tgt = sample_stream_batch(length, batch, rng)
        loss, grads, _ = model.loss_and_grad(seq, tgt)
        model.apply_grad(grads, clip=clip)
    return model


def eval_stream(model, length, n_rows=500, seed=12345):
    rng = np.random.RandomState(seed)
    correct = 0
    bs = 64
    for i in range(0, n_rows, bs):
        b = min(bs, n_rows - i)
        seq, tgt = sample_stream_batch(length, b, rng)
        logits, _ = model.forward_cache(seq)
        correct += int((logits.argmax(1) == tgt).sum())
    return correct / n_rows
