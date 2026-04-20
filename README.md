# MarketForge

So this is a little project I've been working on for fun. I don't really know a whole lot about coding or any of this stuff if I'm being honest, but I wanted to build something and this is the best I could do with what I know. I'm pretty proud of how it turned out though.

The basic idea is — I wanted a way to make fake crypto price data that actually looks and feels like real crypto price data. Like not just random noise, but something that actually copies the patterns and the way prices move. I use it to train machine learning models when I don't have enough real data to work with.

I did my best to write everything as clean as I could and explain how it works below. If something looks weird or could be done better, I'm always open to hearing it. Hope it's useful to someone!

---

Synthetic cryptocurrency market data generator. Downloads real OHLCV data from Binance, generates statistically faithful synthetic candles using IAAFT (Iterative Amplitude Adjusted Fourier Transform), then scores the output across six quantitative metrics and an ML benchmark.

---

## How It Works

The pipeline has four stages, all orchestrated from a single interactive CLI:

**Stage 1 — Data Ingestion**  
Downloads historical OHLCV from `data.binance.vision` (bulk monthly ZIPs), fills recent gaps with daily ZIPs, then hits the Binance REST API for anything still missing. Data is cached on disk — re-running the same pair/timeframe skips the download entirely. After fetching, a statistical fingerprint is computed: mean/std/skew/kurtosis of log returns, 20-lag ACF of both raw and absolute returns, GARCH(1,1) parameters, Hill tail index, and basic OHLCV structure metrics.

**Stage 2 — Generation (IAAFT Engine)**  
The generator computes log returns from the source data, then produces synthetic chunks via IAAFT:

1. FFT of reference log-returns to extract the amplitude spectrum
2. Phase randomization — new random phases are applied while the amplitude spectrum is held exactly fixed
3. IFFT to reconstruct a surrogate return series
4. Amplitude mapping — the surrogate is rank-sorted to exactly match the original distribution (100% distribution fidelity by construction)
5. Volatility density forcing — absolute returns are re-ranked to match the original volatility clustering pattern
6. Final moments correction — a second rank remap locks the distribution again after step 5 perturbs it

If the requested output length exceeds the source length, the process repeats in chunks and concatenates. OHLCV candles are then reconstructed: close prices from cumulative sum of returns starting at 40000, opens from lagged closes, highs/lows scaled by local volatility.

**Stage 3 — Validation**  
Six metrics, each scored 0–100%, combined into a weighted overall score:

| Metric | Method | Weight |
|---|---|---|
| Distribution | Kolmogorov-Smirnov two-sample test | 15% |
| Moments | Hybrid abs/relative error on mean, std, skew, kurtosis | 15% |
| Autocorrelation | RMSE between ACF curves of raw and abs returns (20 lags) | 20% |
| Volatility Dynamics | GARCH(1,1) persistence, alpha/beta ratio, unconditional variance | 20% |
| Tail Behavior | Hill tail index estimator (top 5% of abs returns) | 10% |
| MMD | RBF kernel MMD with median bandwidth heuristic, calibrated against real-vs-real baseline | 20% |

Graded A/B/C/D/F (90/80/70/60 thresholds). Results saved as both a text report and `validation_scores.json`.

**Stage 4 — TSTR Benchmark**  
Trains an XGBoost classifier to predict next-candle direction (up/down) using lagged returns, MA ratios, rolling volatility, RSI-like momentum, and volume change rate. Runs three experiments:

- **TRTR** — Train on Real, Test on Real (baseline upper bound)
- **TSTR** — Train on Synthetic only, Test on Real (quality test)
- **TSTR+R** — Train on Synthetic + Real combined, Test on Real (augmentation test)

Reports accuracy, F1, and Sharpe ratio for each. The key number is the gap between TSTR and TRTR — under 5% is considered solid.

**Stage 5 — Structural Geometry Validation** (bonus stage beyond the progress bar)  
Detects order blocks (price zones preceding ATR-scaled impulse moves, checks how often price returns to mitigate them) and trendlines (pivot-based multi-touch support/resistance lines) on both real and synthetic data. Compares frequency, mitigation rates, and slope — real BTC typically prints ~14 order blocks per 1000 candles at ~59% mitigation rate. If synthetic data shows 0, the output is statistically smooth but lacks market microstructure.

---

## Project Structure

```
MarketForge/
├── run.py               # Interactive CLI — start here
├── config.py            # All paths, hyperparameters, and helper functions
├── data_ingestion.py    # Binance OHLCV fetcher + statistical fingerprint
├── a_tier_engine.py     # IAAFT generation engine
├── generator.py         # Thin wrapper that calls ATierEngine and saves output
├── validator.py         # Validation metrics, TSTR benchmark, structural analysis
├── requirements.txt     # Dependencies
├── input_data/
│   ├── raw_ohlcv/       # Cached real OHLCV CSVs (auto-created)
│   └── fingerprints/    # Statistical fingerprints as JSON (auto-created)
├── generated_data/      # Synthetic output, one folder per run (auto-created)
└── reports/             # Validation and benchmark reports (auto-created)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `pandas`, `scipy`, `statsmodels`, `arch`, `scikit-learn`, `xgboost`, `requests`, `tqdm`

---

## Usage

```bash
python run.py
```

The CLI walks through four prompts:

1. **Pair** — Binance trading pair (e.g. `BTCUSDT`, `ETHUSDT`). Defaults to `BTCUSDT`.
2. **Timespan** — Date range for source data. Format: `[2020-01-01]-[now]`. Data available from ~Aug 2017 for most major pairs.
3. **Timeframe** — Candle interval. Options: `1m 3m 5m 15m 30m 1h 2h 4h 6h 8h 12h 1d`. Defaults to `1h`.
4. **Years to generate** — How many years of synthetic data to produce. Defaults to `10`.

After confirming the config summary, the full pipeline runs automatically. Outputs land in `generated_data/run_NNN_YYYYMMDD_HHMM/a_tier/` and `reports/run_NNN_.../`.

**Run ingestion only:**
```bash
python data_ingestion.py --pair BTCUSDT --timeframe 1h --start 2020-01-01 --end now
```

**Run validation only:**
```bash
python validator.py --real input_data/raw_ohlcv/BTCUSDT_1h.csv --generated generated_data/run_001_.../
```

---

## Output Format

Generated CSVs have columns: `open_time` (ms epoch), `open`, `high`, `low`, `close`, `volume`. Same schema as Binance kline data, directly compatible with most backtesting and ML pipelines.

Run IDs are auto-incremented: `run_001_20240410_1430`, `run_002_...`, etc.

---

## Configuration

All tuneable parameters are in `config.py`:

| Parameter | Default | Description |
|---|---|---|
| `BOOTSTRAP_BLOCK_SIZE` | 50 | Candles per bootstrap block (unused in current engine) |
| `HMM_N_STATES` | 3 | Hidden market states for HMM (unused in current engine) |
| `DIFFUSION_WINDOW_SIZE` | 64 | DDPM training window (unused in current engine) |
| `VALIDATION_ACF_LAGS` | 20 | ACF lags compared during validation |
| `MMD_KERNEL_BANDWIDTH` | 1.0 | Fallback RBF bandwidth (overridden by median heuristic) |
| `TSTR_TRAIN_RATIO` | 0.8 | Train/test split for real data in TSTR benchmark |
| `VALIDATION_WEIGHTS` | see config | Per-metric weights for overall score (must sum to 1.0) |

---

## Notes

The codebase references additional engine types (`tda`, `transformer`, `diffusion`, `agent`) in the validator and benchmark — these were planned or previously implemented engines. Only the IAAFT engine (`a_tier`) is active in the current version.

The IAAFT method guarantees exact distribution matching by construction (rank-sorting forces the output to use the exact same return values as the input). The tradeoff is that it cannot generate returns outside the range observed in the source data — extreme tail events from the training period set a hard ceiling on synthetic tails.

For longer generation runs the source data is reused in chunks with independent phase randomization each time, so adjacent chunks share distribution but not temporal structure at the seam. For most ML training use cases this is fine; for regime-sensitive strategies it may be worth validating the seam behavior directly.
