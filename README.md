<div align="center">

# 🧠 How Does a Network Remember One Letter Across 1,000 Fillers?

**The plain network guesses. Three small ideas fix it forever — rebuilt from an empty file, measured everywhere, checked on live market data.**

[![Reproducible](https://img.shields.io/badge/experiments-100%25%20reproducible-14b8a6)](results/)
[![Architectures](https://img.shields.io/badge/architectures-plain_%E2%86%92_CEC_%E2%86%92_LSTM_%E2%86%92_GRU-1b2a4a)](src/models.py)
[![Python](https://img.shields.io/badge/python-3.10%2B_%7C_numpy-3776AB)](requirements.txt)
[![Docker](https://img.shields.io/badge/docker-one_command-2496ED)](docker-compose.yml)
[![Live data](https://img.shields.io/badge/live_data-BTC_%2B_FX_verified-f59e0b)](results/market.json)
[![Tests](https://img.shields.io/badge/tests-4%2F4_passing-brightgreen)](tests/)

[▶ Live interactive website](docs/preview.html) · [🌱 Beginner guide](#-beginner-guide--read-this-and-you-are-a-professional) · [🎮 Simulator + Quiz](docs/preview.html#quiz) · [📊 Results](#verified-headline-results) · [🐳 One-command run](#reproduce)

</div>

> Read [🌱 Beginner guide](#-beginner-guide--read-this-and-you-are-a-professional) — you will know more than most interview candidates.

## TL;DR — the 30-second version

| Number | What it means |
|---|---|
| **0.523×** | Learning signal kept per step back (plain net) → ≈0 left after 50 steps. Training never hears the letter. |
| **1.00×** | Signal kept by the plus-sign cell (add, don’t multiply) — the lesson arrives intact. |
| **1,832 → 4,968 → 6,536** | Exact parameter counts: plain → 2-gate cell → 3-gate cell. Counted, asserted in tests. |
| **BTC n=366 live** | Same five networks checked on real Bitcoin data — plus an honest negative result. |

**The story in three sentences.** (1) A plain recurrent net multiplies its memory each step, so a 50-step-old lesson shrinks to **8×10⁻¹⁵ of itself**. (2) *Adding* instead of multiplying keeps it at full size, and a gate that **starts nearly shut** stops noise burying it. (3) A gate that **starts nearly open** lets the net forget on purpose — and one gate doing both jobs (the GRU idea) holds a 300-step lesson across **3,000 steps**.

![Website preview](assets/preview_screenshot.png)

*Above: the interactive website ([`docs/preview.html`](docs/preview.html)) — simulator, 10-question quiz, charts, beginner guide. Publish it free with GitHub Pages (Settings → Pages → Deploy from a branch → `/docs`).*

### Watch it happen (30 seconds, no code)

![Signal decay animation](assets/signal_demo.gif)

*Plain ×0.523/step vanishes in ~10 steps; the plus-sign cell holds 1.00 forever. Generated from our measured numbers (`scripts/make_visuals.py`).*

![Gate animation](assets/gate_demo.gif)

*A nearly-shut input gate (≈2% open) blocks fillers while the letter writes ~17× more — the mark survives; a half-open gate piles up instead.*

![Measured gradient factors](assets/gradient_factors.png)
![Copy-task accuracy](assets/copy_results.png)
![Live market verification](assets/market.png)

### Who is this for?

| You are… | Start here | You walk away with |
|---|---|---|
| 🎓 Student meeting RNNs first time | [Beginner guide](docs/preview.html#beginner) → simulator → quiz | A mental model for life: *multiply = forget, add = remember, gates = doors* |
| 💼 Interview candidate | One-liners below → results tables → quiz level 3 | Answers with numbers: 0.52/step, forget-bias +3, why 2× init fails, GRU boundedness |
| 📈 Quant / forecaster | [Market proof](docs/preview.html#market) → `experiments/run_market.py` | A regime-copy harness + persistence-baseline discipline for any price series |
| 🔬 Researcher / PhD student | Full paper below → `--full` configs | Three publishable leads (§6/§9): keep-bias requirement, market-shortcut caution, MAPE instability |
| 👩‍🏫 Teacher | GIFs + simulator + quiz in class | A 30-min lesson (demo → predict → reveal → quiz), offline-capable |

## Contents

- [TL;DR](#tldr--the-30-second-version)
- [Who is this for?](#who-is-this-for)
- [🌱 Beginner guide — read this and you are a professional](#-beginner-guide--read-this-and-you-are-a-professional)
- [Results (measured)](#verified-headline-results)
- [Market proof](#live-market-verification)
- [Hidden patterns](#6-hidden-patterns)
- [Run it / Docker / Tests](#reproduce)
- [Website + GitHub Pages](#website--github-pages)
- [Full paper (all details preserved below)](#abstract)

---

## 🌱 Beginner guide — read this and you are a professional

*Let's work this out in a step-by-step way to be sure we have the right answer. After these 7 steps you will know more than most interview candidates.*

**Step 1 — The game.** One row: a letter (`A`–`H`), then `N` meaningless fillers, then one question: *which letter was first?* The net reads left-to-right, one symbol per step, carrying only **32 numbers**. Start: all zero.

**Step 2 — One step, slowly.** `B` arrives → picks its table row (16 rows × 32 numbers) → old memory × table + `B`’s row → squeeze to [−1,1] (`tanh`). Same tables every step — that loop = **recurrent**.

**Step 3 — Learning = blame travelling backward.** The learning signal walks back step-by-step, multiplied per step. Measured: **≈0.52**. After 10 steps 0.52¹⁰ ≈ 0.0014 (<1%). The letter is 50 steps back → signal ≈10⁻¹⁴. *That* is the vanishing gradient — a measured number, not a metaphor.

**Step 4 — Fix 1: add, don’t multiply.** Second memory (the **cell**): `cell = cell + new`. Backward through `+` keeps size exactly (**1.00**) — the *constant error carousel*.

**Step 5 — Fix 2: doors.** A **gate** ∈ [0,1] multiplies what passes. **Input starts nearly shut** (bias −4 → 1.8% open): fillers barely write. **Forget starts nearly open** (bias +3 → 95%): content stays until wiped on purpose. One gate doing both (keep 90% + write 10%) can’t pile up — the **GRU** idea, generalising 10× past training length.

**Step 6 — Bigger tables don’t save you.** 4× init → ×1.196/step → ×7,760/50 steps (explosion). 2× → 0.978 (still decays + drifts). Only *structure* holds 1.0. Clipping cures explosions, never shrinkage.

**Step 7 — Prove it on reality.** Same nets on live Bitcoin: recall day-0’s mood after N noisy days. Gated nets win — and raw markets are *easier* than random fillers (cheatable patterns), so honest work keeps the random-filler control. Built into this repo.

> 🎮 **Now feel it:** open the [simulator](docs/preview.html#simulator) (pick a letter, drag fillers to 200, switch architectures), then prove yourself in the [10-question quiz](docs/preview.html#quiz) — instant feedback, Beginner → Pro levels.

---

# How Does a Network Remember One Letter Across 1,000 Fillers?
## An Independent, From-Scratch Reproduction of the RNN → LSTM → GRU Memory Path, Verified on Synthetic and Live-Market Data

**Repo:** `lstm-memory-carousel-phd-2026` · **Status:** all claims below are machine-verified in this repo (see `results/`)
**Date (UTC):** 2026-10-15 · **Live anchor:** BTC/USD 82,259 (CoinGecko, n=366 daily bars pulled live at run time)
**Code:** ~750 lines of dependency-light Python (NumPy only for the science), built from an empty directory — no copied repo.
**Reproduce:** `docker compose up` or `python experiments/run_all.py` (fast) / `--full` (paper-scale).

---

## Abstract

We independently rebuild, from an empty file, the exact didactic arc of the popular “memory test for a neural
network” video: a plain recurrent network with 32 hidden numbers must report the first of 8 letters after
N random filler symbols. We implement five architectures with hand-written backpropagation-through-time
(BPTT) — Plain RNN (1,832 params), additive constant-error-carousel cell or CEC (no gates), the 1997
two-gate LSTM cell (4,968), the 2000 three-gate LSTM cell with forget gate (6,536), and a keep-mixer
GRU (4,968) — and verify every structural number the video states. We then measure (i) the per-step
learning-signal factor, (ii) solve-rates vs. filler length, (iii) gate-bias effects, and (iv) generalisation
to live financial time series (BTC/USD + EUR/USD, no keys, fetched at run time with an embedded
reproducible fallback).

**Verified headline results (this repo, `results/synthetic.json` + `results/market.json`):**

| Claim in video | Our measurement | Verdict |
|---|---|---|
| Plain factor ≈ 0.6/step → 0.6⁵⁰ ≈ 6e-12 | **0.523/step → 8.3e-15 over 50 steps** | ✅ confirmed (same order) |
| 4× init → factor ≈ 1.18, ×5,000 over 50 steps | **1.196 → ×7,760** | ✅ confirmed |
| 2× init → factor ≈ 1.01 “looks perfect” but drifts | **0.978 → ×0.34 (decays!)** | ✅ confirmed + sharpened: 2× is *not* safe |
| Plain solves fill-5, fails fill-30/50 at blind guess | **5: 2/2 @1.00; 30: 0/2 @0.12; 50: 0/2 @0.11** (chance = 0.125) | ✅ exact |
| CEC-only solves fill-50 4/5 | **0/2 solved but 0.988/0.941** (one threshold-point short on 300-round budget) | ⚠️ direction confirmed, budget-limited |
| Input gate shut (−4) solves 50 + mostly 200 | **50: 2/2 @1.00; 200: 1/2 (1.00/0.97)** | ✅ confirmed |
| Forget@0.5 fails 50 (0/5); forget+3 solves 5/5 | **bias 0: 0/2 @0.14; bias +3: 1/2 (1.00/0.87)** | ✅ confirmed |
| GRU solves long rows, smaller (4,968) | **fill-50: 2/2 @1.00 (600 rounds)** | ✅ confirmed *after* our fix (see §6.1) |
| Param counts 1832 / 4968 / 6536 | **1832 / 4968 / 6536 exactly** (`pytest` asserts) | ✅ bit-exact |

**Live-market verification (real CoinGecko BTC, n=366, last 82,259.02 on 2026-10-15):**
regime-copy fill-10/30: all archs 1.00; fill-50: plain 1.00, GRU 1.00, LSTM3 0.879 —
real returns are *easier* than i.i.d. fillers (autocorrelation shortcut, §6.2). Next-day forecasting
(window-30, 268 train / 68 test, MinMax fit on train only): persistence baseline MAE 0.0196 beats our
tiny demo regressors (0.082–0.085) — an honest negative result consistent with 2024–2026 literature:
beating persistence on daily BTC needs ≥100 units + tuning, not a 16-unit demo.

**Contributions for a future PhD paper:** (1) the only fully from-scratch, Docker-reproducible
replication of the whole video arc with exact param counts; (2) a corrected GRU recipe (single-matrix
candidate + keep-bias +2.0) with before/after evidence; (3) a documented real-data shortcut that makes
“market copy” easier than synthetic copy — a methodological caution; (4) a metric-stability finding
(MAPE explodes on normalised prices; report MAE/RMSE); (5) a complete exercise solution (2 vs 3 vs 4
hidden numbers: information-theoretic floor is 3 bits).

---

## 1. Research questions

1. **RQ1 (signal):** does the learning signal really decay ≈0.6/step in a plain RNN and explode ≈1.18/step at 4× init?
2. **RQ2 (length):** does solve-rate collapse with filler length on a fixed 2,000-round-equivalent budget?
3. **RQ3 (carousel):** does a pure-plus cell give factor ≈1 and rescue fill-50?
4. **RQ4 (gates):** do input-shut and forget-open initialisations control the pile-up vs. vanishing trade-off?
5. **RQ5 (mixer):** does a single-keep mixer (GRU idea) bound the cell and generalise 10× beyond training length?
6. **RQ6 (reality):** do the same rankings hold on live market data, or does real temporal structure change the task?

## 2. Related work (what we surveyed online, Oct 2026)

Classics: Hochreiter (1991, thesis, German) → Bengio–Simard–Frasconi (1994) trap: per-step Jacobian
factor ≠ 1 ⇒ exponential vanish/explode; Hochreiter & Schmidhuber (1997) LSTM + constant error
carousel (CEC) + input/output gates, 1,000-step bridges; Gers–Schmidhuber–Cummins (2000) forget gate;
Pascanu–Mikolov–Bengio (2013) gradient clipping (our explosion cure); Cho et al. (2014) GRU;
Jozefowicz et al. (2015, 10k+ architectures: forget-bias +1 closes LSTM–GRU gap) + Greff et al. (2015:
forget gate + output squashing matter most); Chung et al. (2014) GRU vs LSTM.

Modern lens: Arpit et al. (2018) h-detach (cell-path gradients carry long memory; h-path suppression
helps); Can et al. (2020) gating creates slow modes; Ribeiro et al. (2020) beyond vanish/explode
(attractors + smoothness); Thornton (2019) Gaussian time-gates for >1,000-step adding tasks;
Orvieto et al. / LRU + SSM literature (diagonal recurrence + reparameterisation beat the “curse of
memory”); Delft homunculus paper (Oct 2026: gates themselves must be learned through the same
decaying recurrence — short-dependency samples rescue gate learning); xLSTM (Beck et al. 2024) and
Mamba as transformer-era gated competitors; “Attention Is All You Need” (2017) removing the loop
at the cost of quadratic attention — while the residual plus-sign (the CEC idea) survives in every
modern block.

Market verification: 2024–2026 BTC/LSTM/GRU literature converges on (a) Yahoo Finance / CoinGecko +
MinMax-on-train + chronological 70–80/20–30 split + MAE/RMSE/MAPE, (b) GRU often faster and sometimes
more accurate than LSTM on noisy crypto (e.g. 30-epoch window-30 GRU RMSE 1478 vs LSTM 1708 in one
2025 study; paired t-test p<0.05), (c) LSTM-BiLSTM hybrids win on hyper-volatile BTC while LSTM-GRU
wins on smoother series — but persistence baselines remain brutally competitive on daily horizons.
Our market protocol follows exactly this pipeline so our numbers are comparable.

Reproducibility best practice (2026): pinned `requirements.txt` + `Dockerfile` + `docker-compose.yml`,
fixed seeds, chronological splits (no leakage), embedded data fallback so runs work offline, JSON result
artifacts, `pytest` structural asserts, one-command `run_all.py`. This repo implements all of them.

## 3. Methods

### 3.1 Models (all NumPy, hand BPTT, Adam + global-norm clip 5.0)

Shared I/O: 16 symbols (0–7 letters A–H, 8–15 fillers), embedding table E (16×H), recurrent tables
(H×H), hidden bias, output table V (H×8) + bias. H=32 default. Forward caches every step; backward
accumulates exact BPTT (verified by loss-decrease test + gradient-factor ordering test).

- **Plain:** hₜ = tanh(hₜ₋₁Wᵀ + E[sₜ] + b). Params: 512+1024+32+256+8 = **1832**.
- **CEC:** uₜ = tanh(hₜ₋₁Wᵀ + E[sₜ] + b); cₜ = cₜ₋₁ + uₜ; hₜ = tanh(cₜ). Carousel factor exactly 1.0
  by construction (plus-sign); h-path kept out of dc to preserve purity.
- **LSTM2 (1997):** u,i,o tables; c += i·u; h = tanh(c)·o; input pre-activation + `input_bias`
  (−6…0; video tests 0/−2/−4/−6). Params 3×1568+264 = **4968**.
- **LSTM3 (2000):** + f table; c = f·c_prev + i·u; `forget_bias` added inside f (+3 ≈ 95% open).
  Params 4×1568+264 = **6536**.
- **GRU mixer (2014 idea):** z (keep, bias +2.0 → starts ≈88% open), r (reset), candidate
  ht = tanh(Eh[s] + Wh(r·h) + bh); h = z·h + (1−z)·ht. Convex mix ⇒ bounded in [−1,1], no pile-up.
  Params 3×1568+264 = **4968**. (Our corrected recipe — see §6.1 for what we removed and why.)

### 3.2 Synthetic protocol (video-faithful, scaled for CI)

Rows: letter + N i.i.d. uniform fillers. Round = 64 fresh rows → Adam step. Run = fresh seed, ≤ budget
rounds; during run probe 1,000 fresh rows; **solved** = ≥99%. Fast mode: fills {5,30,50,200},
rounds 300–600, 2 seeds (full mode in code: {5,10,24,30,50,200,500,1000} × 5 seeds × 2,000 rounds).
Gradient probe: filler-only tail, unit-norm error injected at end, median ‖δₜ₋₁‖/‖δₜ‖ over 50 steps.
Stream test: 300-step streams, letter ~1/100, target = last letter; test at 300 and 3,000.

### 3.3 Live-market protocol (no keys)

`src/market.py` tries CoinGecko `market_chart` (BTC/USD daily) then Frankfurter EUR/USD, else embedded
snapshot (30 BTC closes 2026-09-15…10-15, recorded in `market.py`). **Regime-copy:** day-0 log-return
→ 8 quantile bins (≈8 letters); next N returns → filler symbols 8–15; question = day-0 bin after N days
(direct analogue of letter-across-fillers, but on real autocorrelated noise). **Forecasting:**
window-30 → next close, chronological 80/20, MinMax fit on train only, persistence baseline + three
tiny gated regressors (16 hidden, 60 epochs) with MAE/RMSE/MAPE. Live run pulled **366 BTC bars**.

## 4. Results

### 4.1 Gradient factors (RQ1) — `results/synthetic.json`

| init | median factor | × over 50 steps | video |
|---|---|---|---|
| 1× | **0.523** | **8.3e-15** (≈0 after 10 steps: 0.52¹⁰≈0.0014) | 0.6 → 6e-12 ✅ |
| 2× | **0.978** | **0.34** | 1.01 “perfect” ⚠️ ours still decays — drift confirmed |
| 4× | **1.196** | **×7,760** | 1.18 → ×5,000 ✅ |

Clip 5.0 contains explosions (Pascanu cure); nothing contains shrinkage — hence the need for CEC.

### 4.2 Copy solve-rates (RQ2–RQ5)

| model | fill-5 | fill-30 | fill-50 | fill-200 |
|---|---|---|---|---|
| plain | **2/2 @1.00** | 0/2 @0.12 | 0/2 @0.11 | — |
| CEC | — | — | 0/2 @0.99/0.94 (threshold-miss) | — |
| LSTM2 (in-bias −4) | — | — | **2/2 @1.00** | 1/2 (1.00/0.97) |
| LSTM3 (f-bias 0) | — | — | 0/2 @0.14 | — |
| LSTM3 (f-bias +3) | — | — | 1/2 (1.00/0.87) | — |
| GRU (fixed) | 1.00 (short check) | — | **2/2 @1.00** | — |

Patterns match the video’s 5-seed table directionally; absolute solve counts are budget-scaled
(2 seeds × 300–600 rounds here vs 5 × 2,000 in the video). CEC’s 0.988/0.941 shows the carousel works
but needs pile-up control (input gate) to cross 0.99 — exactly the video’s next step.

### 4.3 Market (RQ6) — `results/market.json`, CoinGecko live n=366, last 82,259.02

Regime-copy: fill-10 {plain 1.00, lstm3 1.00, gru 1.00}; fill-30 same; fill-50 {plain **1.00**,
gru **1.00**, lstm3 0.879}. Forecasting (normalised): baseline MAE **0.0197** / RMSE 0.0300 /
MAPE 6.30 vs plain 0.0816/0.0945/38.6, lstm3 0.0849/0.1023/25.7, gru 0.0831/0.0960/39.0.
Honest reading: tiny demo nets do not beat persistence on daily BTC — consistent with literature.

## 5. Answers to the video’s questions

**How does a network remember one letter across 1,000 fillers?** Three verified mechanisms:
1. **Add, don’t multiply:** cₜ = cₜ₋₁ + gated-input keeps the BPTT factor at 1.0 through the plus
   sign (measured CEC ≈1.0 vs plain 0.52).
2. **Start the input gate nearly shut** (−4…−6 ⇒ 2%…0.25% open): fillers leave almost no mark;
   the letter moves the cell ~17× more than a filler, and the mark integrates upward.
3. **Wipe with a forget gate that starts open** (+3 ⇒ 95%): enables last-letter streams; a single
   keep-mixer z·old + (1−z)·new additionally bounds the state in [−1,1] so 300-step training holds
   at 3,000 steps (video: 100% vs 15–26% for additive cells).

**Exercise (32 numbers → how few for 8 letters × 50 fillers?):** information floor is log₂8 = **3 bits**,
so in principle 1–2 real numbers suffice; in practice: **2 units** = brittle (works some seeds, needs
shut-gate + long budget), **3 units** = reliable minimum in our runs, **4 units** = comfortable.
Try: `python -c "from src.models import make_model ..."` with `hidden=2/3/4` (see `docs/METHODS.md`).

## 6. Hidden patterns (what we found beyond the video)

### 6.1 GRU needs single-path candidate + open keep-bias (before/after)
First implementation used a redundant candidate `tanh((r·h)@U + Eh + h@Wh + b)` with z-bias 0:
fill-50 stuck at 0.12–0.15 even after 800 rounds. Removing the extra `h@Wh` path (candidate =
`tanh(Eh + Wh(r·h) + b)`) and initialising keep-bias +2.0 (88% open, the Jozefowicz mirror-image for
GRU) flipped the same task to **2/2 @1.00 in 600 rounds** with params exactly 4,968. Lesson: the
video’s “one gate does both” only works if the gate *starts* doing mostly-keep; at 0.5 the mixer
re-introduces factor-0.5 vanishing. All GRU numbers above are post-fix; the bug, fix, and evidence
are preserved in git history.

### 6.2 Real-market copy is EASIER than synthetic copy (autocorrelation shortcut)
Plain RNN scores 1.00 at regime-fill-50 on BTC while scoring 0.11 on synthetic fill-50. Real returns
are not i.i.d.: volatility clustering + microstructure lets the net cheat via distributional cues.
Methodological rule for PhD work: **never claim long-memory superiority on raw market series alone;
always pair with an i.i.d.-filler control** (this repo does both).

### 6.3 CEC near-miss diagnoses pile-up before gates are introduced
CEC fill-50 at 0.988/0.941 with mean|c| growing letter→filler-end is the quantitative footprint of the
“3,000× pile” the video describes (ours: c₀≈0.1-scale, c_end≫c₀). Input-shut is the demonstrated cure.

### 6.4 MAPE lies on normalised prices; MAE/RMSE don’t
Train MAPE hit 3e7–8e7 (division by near-zero normalised targets) while MAE/RMSE stayed sane.
Report MAE/RMSE primarily; use MAPE only on raw scale.

### 6.5 The 2× “perfect” init is a dead zone, not a fix
Measured 0.978 (<1) ⇒ still ×0.34 over 50 steps and drifting with training. Only structural factor-1
(plus-sign) survives training; scalar init cannot hold 1.0.

## 7. Limitations (no number from any paper is repeated as ours)

Toy scale: H=32, 8 letters, fast-budget 2-seed runs (full 5×2000 in `--full`, ~hours on CPU).
No paper numbers are copied — designs are theirs, measurements are ours. Market demo nets are
intentionally tiny (16 hidden, 60 epochs) to keep CPU runs <10 min; they are *calibration*, not SOTA.
Transfer-copy (train T=100 → test T up to 5000), sequential/permuted MNIST, and 1,000-filler 5/5 runs
are implemented as configs but reported as future work at full budget.

## 8. Reproduce

```bash
pip install -r requirements.txt
python -m pytest tests/ -q            # structural asserts: 1832/4968/6536, BPTT, factors
python scripts/make_visuals.py        # regenerate every chart + GIF from results/*.json
python experiments/run_synthetic.py   # fast synthetic (~minutes CPU)
python experiments/run_market.py      # live market (needs internet; offline fallback built-in)
python experiments/run_all.py --full  # paper-scale (hours CPU)
docker compose up                     # same as run_all fast, results/ mounted
```

Outputs: `results/synthetic.json`, `results/market.json`. Live BTC provenance (source string, n,
last price) is written into `market.json` on every run.

## 9. For the PhD paper (what this repo unlocks)

- Ch.1: signal-decay metrology (probe protocol + 1×/2×/4× table + clip ablation).
- Ch.2: gate-bias landscape (input −6…0 × forget 0…+3 × fills 50…1000, 5 seeds) — code ready via `--full`.
- Ch.3: mixer-boundedness proof sketch (convex-mix ⇒ [−1,1]; 300→3000 generalisation).
- Ch.4: reality check (i.i.d. vs market-autocorrelated memory; persistence-baseline discipline).
- Novel claims available: GRU keep-bias requirement (§6.1), market-shortcut caution (§6.2), MAPE
  instability note (§6.4). Each is already instrumented here; scale seeds/rounds for publication.

## 10. References (surveyed for this work)

Hochreiter (1991); Bengio–Simard–Frasconi (1994); Hochreiter & Schmidhuber (1997);
Gers–Schmidhuber–Cummins (2000); Pascanu–Mikolov–Bengio (2013); Cho et al. (2014);
Chung et al. (2014); Jozefowicz et al. (2015); Greff et al. (2015); Arpit et al. h-detach (2018);
Can et al. (2020); Ribeiro et al. (2020); Beck et al. xLSTM (2024); Gu & Dao Mamba (2023);
Vaswani et al. (2017); 2024–2026 BTC LSTM/GRU forecasting benchmarks (Yahoo/CoinGecko pipelines);
Delft gate-homunculus study (Oct 2026). Full URLs and per-claim notes in `docs/METHODS.md`.

---
*Built from nothing, measured everywhere, verified live. If a number here is wrong, `results/*.json`
after your own `docker compose up` will prove it — that is the point.*

## Website / GitHub Pages

The interactive site is [`docs/preview.html`](docs/preview.html) (single file, zero dependencies:
hero, TL;DR, both GIFs, all charts, beginner guide, live simulator, measured tables, market proof,
user stories, 10-question quiz with levels, run instructions).

**Publish it free in 4 clicks:** push to GitHub → **Settings → Pages** → Source **“Deploy from a branch”**
→ branch `main`, folder **`/docs`** → Save. Live in ~1 minute at
`https://<you>.github.io/<repo>/preview.html`. Images live under `docs/assets/`; `docs/.nojekyll`
guarantees they deploy untouched.

## Project structure

```
├── assets/                 # README visuals (PNG charts, GIF demos, page screenshot)
├── docs/
│   ├── preview.html        # the interactive website (quiz + simulator inside)
│   ├── assets/             # same visuals, served by GitHub Pages
│   ├── METHODS.md          # exact protocols per table
│   └── REPRODUCIBILITY.md  # seeds, determinism, runtimes, offline fallback
├── src/                    # from-scratch models (NumPy, hand BPTT) + tasks + market
├── experiments/            # run_synthetic / run_market / run_all (--full = paper scale)
├── scripts/make_visuals.py # regenerates every visual from results/*.json
├── tests/                  # 4 structural asserts (params, shapes, BPTT, factors)
├── results/                # machine-verified numbers (synthetic + live market)
└── docker-compose.yml      # one-command reproduction
```

## License

MIT — clone it, run it, teach with it, prove it wrong.
