# METHODS — exact protocols so a stranger can redo every table

## A. Synthetic copy task
- Alphabet: symbols 0..7 = A..H (letters), 8..15 = fillers. Row = [letter] + N uniform fillers.
- Hidden H=32, zero init. Batch 64/round. Adam lr 2e-3, betas (0.9,0.999), global-norm clip 5.0.
- Solved = accuracy >= 0.99 on 1,000 fresh rows (seed-separated from training RNG).
- Fast grid (CI): plain {5,30,50}@300r; cec {50}@300r; lstm2 bias-4 {50,200}@400r;
  lstm3 {50}@400r (forget 0 and +3); gru {50}@600r. Full grid (--full): fills
  {5,10,24,30,50,200,500,1000} x 5 seeds x 2000 rounds (video-scale, hours CPU).
- Gate-bias convention: value b added inside sigmoid pre-activation. sigmoid(-6)=0.25%,
  (-4)=1.8%, (-2)=11.9%, (0)=50%, (+1)=73%, (+3)=95.3%. PyTorch default LSTM forget ~= 0.5
  (bias 0) — documented failure mode, reproduced here (0/2 @0.14).

## B. Gradient probe
- PlainRNN, filler-only tail T=50, B=32. Forward with seed-fixed weights at wscale 1/2/4.
- Inject unit-norm random delta at h_T, backpropagate analytically (tanh' then @W each step),
  record median_t ||delta_{t-1}||/||delta_t||. Growth_T = factor^T.
- Our seeds: 1x 0.523, 2x 0.978, 4x 1.196 (video: 0.6 / 1.01 / 1.18 — same qualitative ladder).

## C. Market tasks
- BTC daily closes: CoinGecko market_chart (vs USD, daily) at runtime; fallback = 31 embedded
  closes 2026-09-15..10-15 (also recorded in market.py). Provenance (source string, n, last)
  written to results/market.json every run. FX: Frankfurter EURUSD, synthetic-walk fallback.
- Regime-copy: log-returns -> 8 quantile bins; marker bin = 'letter', next N returns -> filler
  symbols 8..15. 300 training rounds x 64 rows resampled from history; test 256 fresh rows.
- Forecasting: window 30 -> next close; chronological 80/20; MinMax fitted on TRAIN only;
  persistence baseline + 3 tiny gated regressors (H=16, 60 epochs, lr 3e-3). Metrics MAE/RMSE/MAPE;
  MAPE reported on normalised scale is unstable (see README §6.4) — trust MAE/RMSE.

## D. Structural asserts (tests/test_models.py)
- n_params: plain 1832, lstm2 4968, lstm3 6536, gru 4968. Forward shapes (B,8) for all archs.
- Loss decreases on fill-5 (20 steps). Factor ordering: 1x < 1.0 and 4x > 1x.

## E. Online-survey provenance (Oct 2026)
- websearch (Exa): LSTM 1000-step + adding-problem + homunculus-2026 + BTC-2026 benchmarks.
- duckduckgo: forget-bias init + copy-task repos. openresearch: copy-task XOR + OpenAlex LSTM/CEC.
- paper-search (arxiv/semantic/openalex/crossref): Bengio-1994/Pascanu-2013/h-detach/GRU-forecasting.
- agent-reach web: GRU-vs-LSTM long-sequence. wiki: LSTM/vanishing-gradient pages.
- kaggle: LSTM/GRU stock-forecast notebooks. gitmcp: PyTorch docs (bias/optimiser practice).
- gsd_websearch + superpowers: Docker reproducibility + eval methodology.
- Live anchors: current date 2026-10-15 (UTC tool), BTC 30d history + 366d market pull (CoinGecko).
