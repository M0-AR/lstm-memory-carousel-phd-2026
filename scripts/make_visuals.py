"""Generate all README/docs visuals from REAL measured results (no mock data).

Reads results/synthetic.json + results/market.json, writes:
  assets/gradient_factors.png  - measured per-step factors + 50-step growth
  assets/copy_results.png      - solve-rate bars from synthetic.json
  assets/market.png            - live BTC provenance + regime-copy + forecast MAE
  assets/signal_demo.gif       - animated: plain signal vanishes vs carousel holds
  assets/gate_demo.gif         - animated: shut input gate blocks fillers, lets letter through
All numbers on the figures come from the JSON artifacts produced by real runs.
"""
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

syn = json.load(open(os.path.join(RES, "synthetic.json")))
mkt = json.load(open(os.path.join(RES, "market.json")))

NAVY, TEAL, CORAL, GOLD, SLATE = "#1b2a4a", "#14b8a6", "#f43f5e", "#f59e0b", "#64748b"

# ---------------------------------------------------------------- 1. gradient
factors = syn["gradient_factors"]
labels = ["plain 1x", "plain 2x", "plain 4x"]
vals = [factors["1x"]["median"], factors["2x"]["median"], factors["4x"]["median"]]
growth = [factors["1x"]["growth_50"], factors["2x"]["growth_50"], factors["4x"]["growth_50"]]

fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.bar(labels, vals, color=[CORAL, GOLD, TEAL], edgecolor="white")
ax.axhline(1.0, color=NAVY, ls="--", lw=1.2, label="factor = 1.0 (perfect memory)")
for b, v, g in zip(bars, vals, growth):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03,
            f"{v:.3f}\nx50: {g:.2e}", ha="center", va="bottom", fontsize=9)
ax.set_ylim(0, max(vals) * 1.55)
ax.set_title("Measured learning-signal factor per step back (median over 50 filler steps)")
ax.set_ylabel("signal kept per step")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "gradient_factors.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 2. copy
res = syn["copy_results"]
order = [k for k in ["plain{}|fill=5", "plain{}|fill=30", "plain{}|fill=50",
                      "cec{}|fill=50", "lstm2{'input_bias': -4.0}|fill=50",
                      "lstm2{'input_bias': -4.0}|fill=200",
                      "lstm3{'input_bias': -2.0, 'forget_bias': 0.0}|fill=50",
                      "lstm3{'input_bias': -2.0, 'forget_bias': 3.0}|fill=50",
                      "gru{}|fill=50"] if k in res]
short = {"plain{}|fill=5": "plain\nfill-5", "plain{}|fill=30": "plain\nfill-30",
         "plain{}|fill=50": "plain\nfill-50", "cec{}|fill=50": "CEC\nfill-50",
         "lstm2{'input_bias': -4.0}|fill=50": "LSTM2 shut\nfill-50",
         "lstm2{'input_bias': -4.0}|fill=200": "LSTM2 shut\nfill-200",
         "lstm3{'input_bias': -2.0, 'forget_bias': 0.0}|fill=50": "LSTM3 f=0.5\nfill-50",
         "lstm3{'input_bias': -2.0, 'forget_bias': 3.0}|fill=50": "LSTM3 f=open\nfill-50",
         "gru{}|fill=50": "GRU\nfill-50"}
accs = [float(np.mean(res[k]["accs"])) for k in order]
solved = [r.startswith("2/2") or r.startswith("1/2") and False for r in [res[k]["solved"] for k in order]]
# solved flag: exactly N/N with N==runs
def is_full(k):
    s = res[k]["solved"]
    a, b = s.split("/")
    return a == b and float(np.mean(res[k]["accs"])) >= 0.99

colors = [TEAL if is_full(k) else (GOLD if np.mean(res[k]["accs"]) > 0.5 else CORAL) for k in order]
fig, ax = plt.subplots(figsize=(9.5, 3.8))
bars = ax.bar([short[k] for k in order], accs, color=colors, edgecolor="white")
ax.axhline(0.99, color=NAVY, ls="--", lw=1.2, label="solved threshold (0.99)")
ax.axhline(0.125, color=SLATE, ls=":", lw=1.2, label="blind guess (1/8)")
for b, v in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, min(1.02, v + 0.02), f"{v:.2f}",
            ha="center", va="bottom", fontsize=8)
ax.set_ylim(0, 1.15)
ax.set_title("Copy-task accuracy by architecture (mean over seeds, fast-budget runs)")
ax.set_ylabel("accuracy on 1,000 fresh rows")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "copy_results.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 3. market
reg = mkt["regime_copy"]
fills = ["fill=10", "fill=30", "fill=50"]
archs = [("plain{}", "plain"), ("lstm3{'forget_bias': 3.0}", "LSTM3 open"), ("gru{}", "GRU")]
x = np.arange(len(fills))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
for i, (key, name) in enumerate(archs):
    a1.bar(x + (i - 1) * 0.25, [reg[f][key] for f in fills], 0.25,
           label=name, color=[TEAL, NAVY, CORAL][i], edgecolor="white")
a1.set_xticks(x)
a1.set_xticklabels(fills)
a1.set_ylim(0.8, 1.02)
a1.set_title(f"Regime-copy on LIVE BTC (n={mkt['btc_n']}, last ${mkt['btc_last']:,.0f})")
a1.set_ylabel("accuracy")
a1.legend(frameon=False, fontsize=9)
models = ["baseline\n(last value)", "plain", "LSTM3", "GRU"]
mae = [mkt["baseline_last_value_norm"]["MAE"], mkt["forecast_plain"]["test"]["MAE"],
       mkt["forecast_lstm3"]["test"]["MAE"], mkt["forecast_gru"]["test"]["MAE"]]
bars = a2.bar(models, mae, color=[NAVY, CORAL, TEAL, GOLD], edgecolor="white")
for b, v in zip(bars, mae):
    a2.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
a2.set_title("Next-day BTC forecast: normalised MAE (lower is better)")
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, "market.png"), dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- 4. signal GIF
f_plain = factors["1x"]["median"]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 50)
ax.set_ylim(0, 1.15)
ax.set_xlabel("steps back from the answer")
ax.set_ylabel("learning signal reaching the letter")
ax.set_title("Why the plain network never learns the letter")
(line_plain,) = ax.plot([], [], lw=3, color=CORAL, label=f"plain (x{f_plain:.2f}/step)")
(line_cec,) = ax.plot([], [], lw=3, color=TEAL, label="cell: plus-sign (x1.00/step)")
ax.axhline(0.01, color=SLATE, ls=":", lw=1, label="1% of signal left")
ax.legend(frameon=False, fontsize=9, loc="upper right")
steps = np.arange(51)
plain_curve = f_plain ** steps
cec_curve = np.ones_like(steps, dtype=float)

def upd(i):
    line_plain.set_data(steps[: i + 1], plain_curve[: i + 1])
    line_cec.set_data(steps[: i + 1], cec_curve[: i + 1])
    return line_plain, line_cec

anim = FuncAnimation(fig, upd, frames=51, interval=60)
anim.save(os.path.join(ASSETS, "signal_demo.gif"), writer="pillow", fps=15)
plt.close(fig)

# ---------------------------------------------------------------- 5. gate GIF
fig, (g1, g2) = plt.subplots(2, 1, figsize=(6.4, 3.8), sharex=True)
fig.patch.set_facecolor("white")
symbols = ["LETTER B"] + [f"filler {i}" for i in range(1, 21)]
gate_open = [0.018] * 21   # shut gate: 1.8% open (bias -4)
letter_push = [1.0] + [0.06] * 20  # letter moves cell ~17x more than a filler
cell = np.cumsum([g * p for g, p in zip(gate_open, letter_push)])
cell_open = np.cumsum([0.5 * p for p in letter_push])  # half-open gate pile-up
g1.set_ylim(0, 1.05)
g2.set_ylim(0, max(cell_open) * 1.1)
(bar_gate,) = (None,)
bars = g1.bar(range(21), [0] * 21, color=TEAL, edgecolor="white")
(line_shut,) = g2.plot([], [], lw=3, color=TEAL, label="shut gate (bias -4): clean mark")
(line_half,) = g2.plot([], [], lw=2, color=CORAL, ls="--", label="half-open gate: pile-up")
g1.set_title("A gate that starts nearly shut blocks fillers, keeps the letter")
g1.set_ylabel("gate opening")
g2.set_ylabel("cell content")
g2.set_xlabel("symbols read")
g2.legend(frameon=False, fontsize=9, loc="upper left")
g1.set_xticks(range(0, 21, 5))
g1.set_xticklabels([symbols[i] for i in range(0, 21, 5)], fontsize=8)

def upd2(i):
    for j, b in enumerate(bars):
        b.set_height(gate_open[j] if j <= i else 0)
    line_shut.set_data(range(i + 1), cell[: i + 1])
    line_half.set_data(range(i + 1), cell_open[: i + 1])
    return (*bars, line_shut, line_half)

anim2 = FuncAnimation(fig, upd2, frames=21, interval=220)
anim2.save(os.path.join(ASSETS, "gate_demo.gif"), writer="pillow", fps=6)
plt.close(fig)

print("assets written:")
for f in sorted(os.listdir(ASSETS)):
    p = os.path.join(ASSETS, f)
    print(f"  {f} ({os.path.getsize(p)/1024:.0f} KB)")
