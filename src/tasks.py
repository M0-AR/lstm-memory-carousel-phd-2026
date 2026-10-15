"""Tasks: synthetic copy-memory rows + long-stream last-letter + helpers."""
import numpy as np

LETTERS = list("ABCDEFGH")
N_LETTER = 8
N_FILL = 8
N_SYM = 16


def sample_copy_batch(n_fill, batch, rng):
    """Row = [letter, filler*n_fill]; target = letter index. Returns seq [T,B], targets [B]."""
    T = 1 + n_fill
    letters = rng.randint(0, N_LETTER, size=batch)
    seq = np.zeros((T, batch), dtype=np.int64)
    seq[0] = letters
    if n_fill:
        seq[1:] = rng.randint(N_LETTER, N_SYM, size=(n_fill, batch))
    return seq, letters.astype(np.int64)


def eval_accuracy(model, n_fill, n_rows=1000, seed=999):
    rng = np.random.RandomState(seed)
    correct = 0
    bs = 128
    for i in range(0, n_rows, bs):
        b = min(bs, n_rows - i)
        seq, tgt = sample_copy_batch(n_fill, b, rng)
        logits, _ = model.forward_cache(seq)
        pred = logits.argmax(axis=1)
        correct += int((pred == tgt).sum())
    return correct / n_rows


def sample_stream_batch(length, batch, rng, p_letter=0.01):
    """Long stream; letters arrive ~1/100 steps; target = LAST letter in stream.
    Fillers = 8..15; letters 0..7 guaranteed at least one (position 0 forced letter)."""
    seq = rng.randint(N_LETTER, N_SYM, size=(length, batch)).astype(np.int64)
    targets = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        # force first symbol a letter so target always defined
        first = rng.randint(0, N_LETTER)
        seq[0, b] = first
        last = first
        for t in range(1, length):
            if rng.rand() < p_letter:
                lv = rng.randint(0, N_LETTER)
                seq[t, b] = lv
                last = lv
        targets[b] = last
    return seq, targets
