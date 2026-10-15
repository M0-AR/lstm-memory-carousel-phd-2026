"""From-scratch recurrent models (NumPy only).

Five architectures from the video:
  1. PlainRNN    h = tanh(E[s] + W h_prev + bh)            ~1832 params @H=32
  2. CECAdditive c += u ; h = tanh(c)                      (constant error carousel)
  3. LSTM2Gate   input+output gates (Hochreiter & Schmidhuber 1997) ~4968
  4. LSTM3Gate   + forget gate (Gers et al. 2000)           ~6536
  5. GRUMixer    keep-gate mixer (Cho et al. 2014 / GRU)    ~4968

Symbols: 0..7 letters A-H, 8..15 fillers. Output: 8 letter scores.
Hand-written BPTT + Adam + global-norm clip (video's explosion cure).
"""

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def dsigmoid(y):
    return y * (1.0 - y)


def dtanh_from_y(y):
    return 1.0 - y * y


class AdamState:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps

    def step(self, params, grads, clip=5.0):
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()) + 1e-12)
        scale = clip / total if total > clip else 1.0
        self.t += 1
        for k in params:
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _softmax_ce_loss(logits, targets):
    B = logits.shape[0]
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    loss = -np.log(p[np.arange(B), targets] + 1e-12).mean()
    dp = p.copy()
    dp[np.arange(B), targets] -= 1.0
    dp /= B
    return loss, dp


class BaseRNN:
    arch = "base"

    def __init__(self, hidden=32, seed=0, lr=1e-3):
        self.H = hidden
        self.rng = np.random.RandomState(seed)
        self.params = {}
        self.opt = None
        self.lr = lr

    def init_adam(self):
        self.opt = AdamState(self.params, lr=self.lr)

    def n_params(self):
        return sum(v.size for v in self.params.values())

    def apply_grad(self, grads, clip=5.0):
        self.opt.step(self.params, grads, clip=clip)

    def loss_and_grad(self, seq, targets):
        logits, cache = self.forward_cache(seq)
        loss, dlog = _softmax_ce_loss(logits, targets)
        grads = self.backward(seq, dlog, cache)
        return loss, grads, logits


class PlainRNN(BaseRNN):
    arch = "plain"

    def __init__(self, hidden=32, seed=0, lr=2e-3, wscale=1.0):
        super().__init__(hidden=hidden, seed=seed, lr=lr)
        H = hidden
        r = self.rng
        self.params["E"] = r.randn(16, H) * 0.1
        self.params["W"] = r.randn(H, H) * (0.1 * wscale)
        self.params["bh"] = np.zeros(H)
        self.params["V"] = r.randn(H, 8) * 0.1
        self.params["by"] = np.zeros(8)
        self.init_adam()

    def forward_cache(self, seq):
        E, W, bh, V, by = (self.params["E"], self.params["W"], self.params["bh"],
                           self.params["V"], self.params["by"])
        h = np.zeros((seq.shape[1], self.H))
        hs = [h.copy()]
        for t in range(seq.shape[0]):
            h = np.tanh(hs[-1] @ W.T + E[seq[t]] + bh)
            hs.append(h.copy())
        return hs[-1] @ V + by, {"hs": hs}

    def backward(self, seq, dlog, cache):
        E, W = self.params["E"], self.params["W"]
        hs = cache["hs"]
        T = seq.shape[0]
        gE = np.zeros_like(E)
        gW = np.zeros_like(W)
        gbh = np.zeros_like(self.params["bh"])
        gV = hs[-1].T @ dlog
        gby = dlog.sum(axis=0)
        dh = dlog @ self.params["V"].T
        for t in reversed(range(T)):
            dt = dh * dtanh_from_y(hs[t + 1])
            gW += dt.T @ hs[t]
            gbh += dt.sum(axis=0)
            np.add.at(gE, seq[t], dt)
            dh = dt @ W
        return {"E": gE, "W": gW, "bh": gbh, "V": gV, "by": gby}


class CECAdditive(BaseRNN):
    """Cell only adds: c_t = c_{t-1} + tanh(E[s] + W h_{t-1} + b); h = tanh(c)."""
    arch = "cec"

    def __init__(self, hidden=32, seed=0, lr=2e-3):
        super().__init__(hidden=hidden, seed=seed, lr=lr)
        H = hidden
        r = self.rng
        self.params["E"] = r.randn(16, H) * 0.1
        self.params["W"] = r.randn(H, H) * 0.1
        self.params["bh"] = np.zeros(H)
        self.params["V"] = r.randn(H, 8) * 0.1
        self.params["by"] = np.zeros(8)
        self.init_adam()

    def forward_cache(self, seq):
        E, W, bh, V, by = (self.params["E"], self.params["W"], self.params["bh"],
                           self.params["V"], self.params["by"])
        h = np.zeros((seq.shape[1], self.H))
        c = np.zeros((seq.shape[1], self.H))
        HS, CS, US = [h.copy()], [c.copy()], []
        for t in range(seq.shape[0]):
            u = np.tanh(HS[-1] @ W.T + E[seq[t]] + bh)
            c = CS[-1] + u
            h = np.tanh(c)
            HS.append(h.copy())
            CS.append(c.copy())
            US.append(u)
        return h @ V + by, {"HS": HS, "CS": CS, "US": US}

    def backward(self, seq, dlog, cache):
        E, W = self.params["E"], self.params["W"]
        HS, CS, US = cache["HS"], cache["CS"], cache["US"]
        T = seq.shape[0]
        gE = np.zeros_like(E)
        gW = np.zeros_like(W)
        gbh = np.zeros_like(self.params["bh"])
        gV = HS[-1].T @ dlog
        gby = dlog.sum(axis=0)
        dh_next = dlog @ self.params["V"].T  # dL/dh_T
        # dL/dc_T via h_T = tanh(c_T)
        dc = dh_next * dtanh_from_y(HS[-1])
        # dL/dh_T already consumed into dc; remaining dh signal flows via W from u_{T}
        dh = dh_next
        for t in reversed(range(T)):
            # u_t contributes to c_t (=c_{t-1}+u_t); dL/du_t = dc * tanh'(pre) = dc*tanh'(u_t)
            du = dc * dtanh_from_y(US[t])
            gW += du.T @ HS[t]
            gbh += du.sum(axis=0)
            np.add.at(gE, seq[t], du)
            dh_prev = du @ W
            if t > 0:
                # h_t (=HS[t]) feeds u_t only; c path: dc_{t-1} = dc_t (plus-sign, factor 1)
                # h_t itself = tanh(c_t_prev) so h-path adds dh_prev*tanh'(c) into dc? No:
                # dh_prev IS dL/dh_{t-1}... wait HS[t] is h_{t} (0-indexed: HS[0]=h_0, HS[t]=h_t).
                # u_{t} used h_t? No: u_t used HS[t] (=h before step t). du@W is dL/d h_before.
                # h_before = tanh(c_before) so part of dh_prev should fold into dc_{t-1}:
                dc_h = dh_prev * dtanh_from_y(HS[t])
                # dc_{t-1} = dc_t (carousel) + (h-path? h_{t-1}=tanh(c_{t-1}) is separate)
                # The h-path gradient w.r.t. c_{t-1} is dc_h, but dh_prev also continues as h-signal.
                # Keep carousel pure: dc stays, and dh carries h-signal:
                dc = dc + dc_h * 0.0  # pure-plus carousel: factor exactly 1.0
                dh = dh_prev
            else:
                dh = dh_prev
        return {"E": gE, "W": gW, "bh": gbh, "V": gV, "by": gby}


def _gate_params(rng, H):
    return rng.randn(16, H) * 0.1, rng.randn(H, H) * 0.1, np.zeros(H)


class LSTM2Gate(BaseRNN):
    """1997 cell: c += i*u ; h = tanh(c)*o. input_bias added inside input gate."""
    arch = "lstm2"

    def __init__(self, hidden=32, seed=0, lr=2e-3, input_bias=-2.0):
        super().__init__(hidden=hidden, seed=seed, lr=lr)
        H = hidden
        r = self.rng
        self.input_bias = float(input_bias)
        for name in ("u", "i", "o"):
            E, W, b = _gate_params(r, H)
            self.params[f"E{name}"] = E
            self.params[f"W{name}"] = W
            self.params[f"b{name}"] = b
        self.params["V"] = r.randn(H, 8) * 0.1
        self.params["by"] = np.zeros(8)
        self.init_adam()

    def forward_cache(self, seq):
        P = self.params
        h = np.zeros((seq.shape[1], self.H))
        c = np.zeros((seq.shape[1], self.H))
        HS, CS, US, IS, OS = [h.copy()], [c.copy()], [], [], []
        for t in range(seq.shape[0]):
            s = seq[t]
            u = np.tanh(h @ P["Wu"].T + P["Eu"][s] + P["bu"])
            i = sigmoid(h @ P["Wi"].T + P["Ei"][s] + P["bi"] + self.input_bias).astype(np.float64)
            o = sigmoid(h @ P["Wo"].T + P["Eo"][s] + P["bo"]).astype(np.float64)
            c = CS[-1] + i * u
            h = np.tanh(c) * o
            HS.append(h.copy()); CS.append(c.copy())
            US.append(u); IS.append(i); OS.append(o)
        return h @ P["V"] + P["by"], {"HS": HS, "CS": CS, "US": US, "IS": IS, "OS": OS}

    def backward(self, seq, dlog, cache):
        P = self.params
        HS, CS, US, IS, OS = (cache["HS"], cache["CS"], cache["US"], cache["IS"], cache["OS"])
        T = seq.shape[0]
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        grads["V"] = HS[-1].T @ dlog
        grads["by"] = dlog.sum(axis=0)
        dh = dlog @ P["V"].T
        dc = np.zeros_like(dh)
        for t in reversed(range(T)):
            c = CS[t + 1]
            u, i, o = US[t], IS[t], OS[t]
            th = np.tanh(c)
            dth = dh * o
            do = dh * th
            dc = dc + dth * dtanh_from_y(th)
            di = dc * u
            du = dc * i
            dpo = do * (o * (1 - o))
            dpu = du * dtanh_from_y(u)
            dpi = di * (i * (1 - i))
            grads["Wu"] += dpu.T @ HS[t]; grads["bu"] += dpu.sum(0)
            grads["Wi"] += dpi.T @ HS[t]; grads["bi"] += dpi.sum(0)
            grads["Wo"] += dpo.T @ HS[t]; grads["bo"] += dpo.sum(0)
            s = seq[t]
            np.add.at(grads["Eu"], s, dpu)
            np.add.at(grads["Ei"], s, dpi)
            np.add.at(grads["Eo"], s, dpo)
            dh = dpu @ P["Wu"] + dpi @ P["Wi"] + dpo @ P["Wo"]
        return grads


class LSTM3Gate(BaseRNN):
    """2000 cell: c = f*c_prev + i*u ; h = tanh(c)*o. forget_bias added inside forget gate."""
    arch = "lstm3"

    def __init__(self, hidden=32, seed=0, lr=2e-3, input_bias=-2.0, forget_bias=0.0):
        super().__init__(hidden=hidden, seed=seed, lr=lr)
        H = hidden
        r = self.rng
        self.input_bias = float(input_bias)
        self.forget_bias = float(forget_bias)
        for name in ("u", "i", "f", "o"):
            E, W, b = _gate_params(r, H)
            self.params[f"E{name}"] = E
            self.params[f"W{name}"] = W
            self.params[f"b{name}"] = b
        self.params["V"] = r.randn(H, 8) * 0.1
        self.params["by"] = np.zeros(8)
        self.init_adam()

    def forward_cache(self, seq):
        P = self.params
        h = np.zeros((seq.shape[1], self.H))
        c = np.zeros((seq.shape[1], self.H))
        HS, CS, US, IS, FS, OS = [h.copy()], [c.copy()], [], [], [], []
        for t in range(seq.shape[0]):
            s = seq[t]
            u = np.tanh(h @ P["Wu"].T + P["Eu"][s] + P["bu"])
            i = sigmoid(h @ P["Wi"].T + P["Ei"][s] + P["bi"] + self.input_bias).astype(np.float64)
            f = sigmoid(h @ P["Wf"].T + P["Ef"][s] + P["bf"] + self.forget_bias).astype(np.float64)
            o = sigmoid(h @ P["Wo"].T + P["Eo"][s] + P["bo"]).astype(np.float64)
            c = f * CS[-1] + i * u
            h = np.tanh(c) * o
            HS.append(h.copy()); CS.append(c.copy())
            US.append(u); IS.append(i); FS.append(f); OS.append(o)
        return h @ P["V"] + P["by"], {"HS": HS, "CS": CS, "US": US, "IS": IS, "FS": FS, "OS": OS}

    def backward(self, seq, dlog, cache):
        P = self.params
        HS, CS, US, IS, FS, OS = (cache["HS"], cache["CS"], cache["US"],
                                 cache["IS"], cache["FS"], cache["OS"])
        T = seq.shape[0]
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        grads["V"] = HS[-1].T @ dlog
        grads["by"] = dlog.sum(axis=0)
        dh = dlog @ P["V"].T
        dc = np.zeros_like(dh)
        for t in reversed(range(T)):
            c = CS[t + 1]
            c_prev = CS[t]
            u, i, f, o = US[t], IS[t], FS[t], OS[t]
            th = np.tanh(c)
            dth = dh * o
            do = dh * th
            dc = dc + dth * dtanh_from_y(th)
            di = dc * u
            du = dc * i
            df = dc * c_prev
            dc = dc * f  # carousel multiplied by forget gate
            dpo = do * (o * (1 - o))
            dpu = du * dtanh_from_y(u)
            dpi = di * (i * (1 - i))
            dpf = df * (f * (1 - f))
            grads["Wu"] += dpu.T @ HS[t]; grads["bu"] += dpu.sum(0)
            grads["Wi"] += dpi.T @ HS[t]; grads["bi"] += dpi.sum(0)
            grads["Wf"] += dpf.T @ HS[t]; grads["bf"] += dpf.sum(0)
            grads["Wo"] += dpo.T @ HS[t]; grads["bo"] += dpo.sum(0)
            s = seq[t]
            np.add.at(grads["Eu"], s, dpu)
            np.add.at(grads["Ei"], s, dpi)
            np.add.at(grads["Ef"], s, dpf)
            np.add.at(grads["Eo"], s, dpo)
            dh = dpu @ P["Wu"] + dpi @ P["Wi"] + dpf @ P["Wf"] + dpo @ P["Wo"]
        return grads


class GRUMixer(BaseRNN):
    """GRU-style keep mixer (Cho et al. 2014, simplified to our symbol-table I/O):
    z = sigmoid(Ez[s] + Wz h_prev + bz + z_bias)   # keep gate, biased open
    r = sigmoid(Er[s] + Wr h_prev + br)            # reset gate
    ht = tanh(Eh[s] + Wh (r*h_prev) + bh)          # candidate
    h = z*h_prev + (1-z)*ht                        # convex mixer: bounded, no pile-up
    z_bias mirrors the forget-bias fix (Jozefowicz et al. 2015): start keep open."""
    arch = "gru"

    def __init__(self, hidden=32, seed=0, lr=2e-3, z_bias=2.0):
        super().__init__(hidden=hidden, seed=seed, lr=lr)
        H = hidden
        r = self.rng
        self.z_bias = float(z_bias)
        for name in ("h", "z", "r"):
            E, W, b = _gate_params(r, H)
            self.params[f"E{name}"] = E
            self.params[f"W{name}"] = W
            self.params[f"b{name}"] = b
        self.params["V"] = r.randn(H, 8) * 0.1
        self.params["by"] = np.zeros(8)
        self.init_adam()

    def forward_cache(self, seq):
        P = self.params
        h = np.zeros((seq.shape[1], self.H))
        HS, ZS, RS, HS_T = [h.copy()], [], [], []
        for t in range(seq.shape[0]):
            s = seq[t]
            z = sigmoid(h @ P["Wz"].T + P["Ez"][s] + P["bz"] + self.z_bias).astype(np.float64)
            rr = sigmoid(h @ P["Wr"].T + P["Er"][s] + P["br"]).astype(np.float64)
            ht = np.tanh(P["Eh"][s] + (rr * h) @ P["Wh"].T + P["bh"])
            h = z * h + (1 - z) * ht
            HS.append(h.copy()); ZS.append(z); RS.append(rr); HS_T.append(ht)
        return h @ P["V"] + P["by"], {"HS": HS, "ZS": ZS, "RS": RS, "HT": HS_T}

    def backward(self, seq, dlog, cache):
        P = self.params
        HS, ZS, RS, HT = cache["HS"], cache["ZS"], cache["RS"], cache["HT"]
        T = seq.shape[0]
        grads = {k: np.zeros_like(v) for k, v in P.items()}
        grads["V"] = HS[-1].T @ dlog
        grads["by"] = dlog.sum(axis=0)
        dh = dlog @ P["V"].T
        for t in reversed(range(T)):
            h_prev = HS[t]
            z, rr, ht = ZS[t], RS[t], HT[t]
            dht = dh * (1 - z)
            dz = dh * (h_prev - ht)
            dh_keep = dh * z
            dpre = dht * dtanh_from_y(ht)
            grads["Wh"] += dpre.T @ (rr * h_prev)
            grads["bh"] += dpre.sum(0)
            dl_dq = dpre @ P["Wh"]  # dL/d(rr*h_prev)
            dl_drr = dl_dq * h_prev
            dpr = dl_drr * (rr * (1 - rr))
            dh_q = dl_dq * rr
            dh_cand = dpre @ P["Wh"] * 0.0  # folded into dh_q; kept for clarity
            dpz = dz * (z * (1 - z))
            grads["Wz"] += dpz.T @ h_prev; grads["bz"] += dpz.sum(0)
            grads["Wr"] += dpr.T @ h_prev; grads["br"] += dpr.sum(0)
            s = seq[t]
            np.add.at(grads["Eh"], s, dpre)
            np.add.at(grads["Ez"], s, dpz)
            np.add.at(grads["Er"], s, dpr)
            dh = dh_keep + dh_q
        return grads


def make_model(arch, hidden=32, seed=0, lr=2e-3, **kw):
    if arch == "plain":
        return PlainRNN(hidden=hidden, seed=seed, lr=lr, wscale=kw.get("wscale", 1.0))
    if arch == "cec":
        return CECAdditive(hidden=hidden, seed=seed, lr=lr)
    if arch == "lstm2":
        return LSTM2Gate(hidden=hidden, seed=seed, lr=lr, input_bias=kw.get("input_bias", -2.0))
    if arch == "lstm3":
        return LSTM3Gate(hidden=hidden, seed=seed, lr=lr,
                         input_bias=kw.get("input_bias", -2.0),
                         forget_bias=kw.get("forget_bias", 0.0))
    if arch == "gru":
        return GRUMixer(hidden=hidden, seed=seed, lr=lr)
    raise ValueError(f"unknown arch {arch}")
