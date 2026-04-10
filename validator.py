"""
validator.py — Multi-metric validation of synthetic market data.

Standalone usage:
    python validator.py --real input_data/raw_ohlcv/BTCUSDT_1h.csv --generated generated_data/run_001_.../

Scores each generated dataset on a 0-100% scale across multiple metrics,
then prints a side-by-side comparison table.
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.spatial.distance import cdist
from statsmodels.tsa.stattools import acf

import config

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL METRICS (each returns 0-100% similarity)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_log_returns(df: pd.DataFrame) -> np.ndarray:
    """Extract log returns from OHLCV DataFrame."""
    closes = df["close"].values.astype(float)
    returns = np.diff(np.log(closes))
    return returns[np.isfinite(returns)]


def score_distribution(real_returns: np.ndarray, synth_returns: np.ndarray) -> float:
    """
    Compare return distributions using the Kolmogorov-Smirnov test.
    
    KS statistic ranges [0, 1]. 0 = identical distributions, 1 = completely different.
    We convert to a percentage similarity.
    """
    max_n = 50000
    if len(real_returns) > max_n:
        real_returns = np.random.choice(real_returns, max_n, replace=False)
    if len(synth_returns) > max_n:
        synth_returns = np.random.choice(synth_returns, max_n, replace=False)
    
    ks_stat, _ = sp_stats.ks_2samp(real_returns, synth_returns)
    return max(0.0, (1.0 - ks_stat) * 100.0)


def score_moments(real_returns: np.ndarray, synth_returns: np.ndarray) -> float:
    """
    Compare statistical moments (mean, std, skewness, kurtosis).
    
    Uses a hybrid absolute-relative error that handles near-zero values correctly.
    For each moment, the error is: |real - synth| / (|real| + tolerance)
    where tolerance prevents division-by-zero for near-zero means.
    """
    # Tolerances calibrated per moment type:
    # - mean: near zero, needs large tolerance relative to std
    # - std: always positive, small tolerance
    # - skew: can be near zero, moderate tolerance
    # - kurtosis: typically 3+ for financial data, moderate tolerance
    real_std = np.std(real_returns)
    
    moments_and_tolerances = [
        (np.mean(real_returns), np.mean(synth_returns), real_std),       # mean vs std scale
        (np.std(real_returns), np.std(synth_returns), real_std * 0.1),   # std
        (sp_stats.skew(real_returns), sp_stats.skew(synth_returns), 0.5),  # skew
        (sp_stats.kurtosis(real_returns), sp_stats.kurtosis(synth_returns), 1.0),  # kurtosis
    ]
    
    errors = []
    for real_val, synth_val, tolerance in moments_and_tolerances:
        denom = abs(real_val) + tolerance
        error = min(abs(real_val - synth_val) / denom, 1.0)
        errors.append(error)
    
    mean_error = np.mean(errors)
    return max(0.0, (1.0 - mean_error) * 100.0)


def score_autocorrelation(real_returns: np.ndarray, synth_returns: np.ndarray, 
                          n_lags: int = config.VALIDATION_ACF_LAGS) -> float:
    """
    Compare autocorrelation functions of returns and absolute returns.
    
    Good synthetic data should reproduce:
    - Near-zero ACF of raw returns (market efficiency)
    - Slowly-decaying positive ACF of absolute returns (volatility clustering)
    """
    acf_real = acf(real_returns, nlags=n_lags, fft=True)
    acf_synth = acf(synth_returns, nlags=n_lags, fft=True)
    
    acf_abs_real = acf(np.abs(real_returns), nlags=n_lags, fft=True)
    acf_abs_synth = acf(np.abs(synth_returns), nlags=n_lags, fft=True)
    
    # RMSE between ACF curves (excluding lag 0 which is always 1.0)
    rmse_returns = np.sqrt(np.mean((acf_real[1:] - acf_synth[1:]) ** 2))
    rmse_abs = np.sqrt(np.mean((acf_abs_real[1:] - acf_abs_synth[1:]) ** 2))
    
    # Combined score (abs returns ACF is more important)
    combined_rmse = 0.3 * rmse_returns + 0.7 * rmse_abs
    
    # Convert to percentage (RMSE typically < 0.5 for decent synthetic data)
    return max(0.0, (1.0 - combined_rmse * 2) * 100.0)


def score_volatility_dynamics(real_returns: np.ndarray, synth_returns: np.ndarray) -> float:
    """
    Compare GARCH(1,1) dynamics fitted to real vs synthetic data.
    
    Compares persistence (alpha+beta), alpha/beta ratio, and unconditional variance.
    Deliberately avoids comparing omega directly since it's numerically unstable.
    """
    try:
        from arch import arch_model
        
        def fit_garch(returns):
            scaled = returns * 100
            model = arch_model(scaled, vol="Garch", p=1, q=1, mean="Constant", rescale=False)
            result = model.fit(disp="off", show_warning=False)
            alpha = result.params.get("alpha[1]", 0)
            beta = result.params.get("beta[1]", 0)
            omega = result.params.get("omega", 0)
            persistence = alpha + beta
            # Unconditional variance: omega / (1 - persistence) if stationary
            uncond_var = omega / (1 - persistence) if persistence < 1 else omega * 10
            return {
                "alpha": alpha,
                "beta": beta,
                "persistence": persistence,
                "uncond_var": uncond_var,
            }
        
        real_params = fit_garch(real_returns)
        synth_params = fit_garch(synth_returns)
        
        # Compare persistence (most important — captures how long volatility shocks last)
        persistence_err = abs(real_params["persistence"] - synth_params["persistence"])
        persistence_score = max(0, 1.0 - persistence_err * 5)  # 0.2 diff = 0% score
        
        # Compare alpha/beta balance (how reactive vs persistent volatility is)
        real_ratio = real_params["alpha"] / max(real_params["beta"], 1e-6)
        synth_ratio = synth_params["alpha"] / max(synth_params["beta"], 1e-6)
        ratio_err = abs(real_ratio - synth_ratio) / (abs(real_ratio) + 0.1)
        ratio_score = max(0, 1.0 - ratio_err)
        
        # Compare unconditional variance (overall volatility level)
        var_err = abs(real_params["uncond_var"] - synth_params["uncond_var"])
        var_denom = abs(real_params["uncond_var"]) + 0.01
        var_score = max(0, 1.0 - var_err / var_denom)
        
        # Weighted combination
        combined = 0.5 * persistence_score + 0.25 * ratio_score + 0.25 * var_score
        return combined * 100.0
    
    except Exception:
        return 50.0


def score_tail_behavior(real_returns: np.ndarray, synth_returns: np.ndarray) -> float:
    """
    Compare tail behavior using the Hill tail index estimator.
    
    Both should have similar fat-tail characteristics.
    """
    def hill_estimator(returns):
        sorted_abs = np.sort(np.abs(returns))[::-1]
        k = max(int(len(sorted_abs) * 0.05), 10)
        top_k = sorted_abs[:k]
        threshold = sorted_abs[k]
        if threshold > 0:
            return 1.0 / np.mean(np.log(top_k / threshold))
        return None
    
    real_hill = hill_estimator(real_returns)
    synth_hill = hill_estimator(synth_returns)
    
    if real_hill is None or synth_hill is None:
        return 50.0
    
    # Use tolerance-based error (Hill index typically 2-5 for financial data)
    error = abs(real_hill - synth_hill) / (abs(real_hill) + 0.5)
    return max(0.0, (1.0 - min(error, 1.0)) * 100.0)


def score_mmd(real_returns: np.ndarray, synth_returns: np.ndarray, 
              bandwidth: float = config.MMD_KERNEL_BANDWIDTH) -> float:
    """
    Maximum Mean Discrepancy with RBF kernel.
    
    A kernel-based statistical distance that captures differences in all
    moments of the distribution. Lower = more similar.
    """
    max_n = 5000
    if len(real_returns) > max_n:
        real_sample = np.random.choice(real_returns, max_n, replace=False)
    else:
        real_sample = real_returns
    
    if len(synth_returns) > max_n:
        synth_sample = np.random.choice(synth_returns, max_n, replace=False)
    else:
        synth_sample = synth_returns
    
    X = real_sample.reshape(-1, 1)
    Y = synth_sample.reshape(-1, 1)
    
    # Auto-set bandwidth using median heuristic
    subset_X = X[np.random.choice(len(X), min(500, len(X)), replace=False)]
    subset_Y = Y[np.random.choice(len(Y), min(500, len(Y)), replace=False)]
    dists = cdist(subset_X, subset_Y, metric="sqeuclidean")
    median_dist = np.median(dists[dists > 0])
    bandwidth = median_dist if median_dist > 0 else bandwidth
    
    def rbf_kernel(A, B, bw):
        sq_dists = cdist(A, B, metric="sqeuclidean")
        return np.exp(-sq_dists / (2 * bw))
    
    K_xx = rbf_kernel(X, X, bandwidth)
    K_yy = rbf_kernel(Y, Y, bandwidth)
    K_xy = rbf_kernel(X, Y, bandwidth)
    
    m, n = len(X), len(Y)
    
    np.fill_diagonal(K_xx, 0)
    np.fill_diagonal(K_yy, 0)
    
    mmd_sq = (K_xx.sum() / (m * (m - 1)) + 
              K_yy.sum() / (n * (n - 1)) - 
              2 * K_xy.sum() / (m * n))
    
    mmd = max(0, mmd_sq) ** 0.5
    
    # Calibrated scaling: compute baseline MMD between two halves of real data
    # to establish what "similar" looks like, then score relative to that
    half = len(real_sample) // 2
    if half > 100:
        X1 = real_sample[:half].reshape(-1, 1)
        X2 = real_sample[half:2*half].reshape(-1, 1)
        K_11 = rbf_kernel(X1, X1, bandwidth)
        K_22 = rbf_kernel(X2, X2, bandwidth)
        K_12 = rbf_kernel(X1, X2, bandwidth)
        h = len(X1)
        np.fill_diagonal(K_11, 0)
        np.fill_diagonal(K_22, 0)
        baseline_mmd_sq = (K_11.sum() / (h * (h-1)) + K_22.sum() / (h * (h-1)) - 2 * K_12.sum() / (h * h))
        baseline_mmd = max(0, baseline_mmd_sq) ** 0.5
        
        # Score: if mmd <= baseline, 100%. Scale down from there.
        if baseline_mmd > 0:
            ratio = mmd / baseline_mmd
            score = max(0, 1.0 - max(0, ratio - 1.0) * 2) * 100.0  # 1.5x baseline = 0%
        else:
            score = max(0.0, (1.0 - min(mmd * 2, 1.0)) * 100.0)
    else:
        score = max(0.0, (1.0 - min(mmd * 2, 1.0)) * 100.0)
    
    return score


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

METRICS = [
    ("Distribution", "distribution", score_distribution),
    ("Moments", "moments", score_moments),
    ("Autocorrelation", "autocorrelation", score_autocorrelation),
    ("Volatility Dynamics", "volatility", score_volatility_dynamics),
    ("Tail Behavior", "tail", score_tail_behavior),
    ("MMD Score", "mmd", score_mmd),
]


def validate_single(real_returns: np.ndarray, synth_returns: np.ndarray) -> dict:
    """Run all validation metrics on one pair of real vs synthetic returns."""
    scores = {}
    for name, key, metric_fn in METRICS:
        scores[key] = metric_fn(real_returns, synth_returns)
    
    # Weighted overall score
    overall = sum(scores[key] * config.VALIDATION_WEIGHTS[key] for key in scores)
    scores["overall"] = overall
    
    return scores


def validate_run(real_csv: str, run_dir: str) -> dict:
    """
    Validate all generated datasets in a run folder against real data.
    """
    real_df = pd.read_csv(real_csv)
    real_returns = _get_log_returns(real_df)
    
    print(f"\n  📊 Source: {os.path.basename(real_csv)} ({len(real_df):,} candles)")
    print(f"  📁 Run:    {os.path.basename(run_dir)}\n")
    
    results = {}
    
    # Search for all method subfolders
    for method_key in ["tda", "transformer", "diffusion", "agent", "a_tier"]:
        method_dir = os.path.join(run_dir, method_key)
        if not os.path.isdir(method_dir):
            continue
        
        csvs = [f for f in os.listdir(method_dir) if f.endswith(".csv")]
        if not csvs:
            continue
        
        synth_path = os.path.join(method_dir, csvs[0])
        synth_df = pd.read_csv(synth_path)
        synth_returns = _get_log_returns(synth_df)
        
        print(f"  ⏳ Scoring {method_key} ({len(synth_df):,} candles)...")
        scores = validate_single(real_returns, synth_returns)
        results[method_key] = scores
    
    if not results:
        print("  ⚠️  No generated data found in run folder!")
        return results
    
    _print_comparison_table(results)
    _print_metric_explanations()
    
    return results


def _print_comparison_table(results: dict):
    """Print a formatted side-by-side comparison table."""
    methods = list(results.keys())
    method_labels = {
        "tda": "TDA Topology",
        "transformer": "MarketGPT",
        "diffusion": "DDPM Diffusion",
        "agent": "Agent Simulation",
        "a_tier": "A-Tier Engine",
    }
    
    col_width = 15
    total_width = 27 + col_width * len(methods)
    
    print()
    print("  ╔" + "═" * (total_width - 2) + "╗")
    title = "VALIDATION RESULTS"
    print(f"  ║{title:^{total_width - 2}}║")
    print("  ╠" + "═" * (total_width - 2) + "╣")
    
    # Header
    header = f"  ║ {'Metric':<24}"
    for m in methods:
        header += f"{method_labels.get(m, m):>{col_width}}"
    header += " ║"
    print(header)
    print("  ╟" + "─" * (total_width - 2) + "╢")
    
    # Metric rows
    for display_name, key, _ in METRICS:
        row = f"  ║ {display_name:<24}"
        for m in methods:
            score = results[m].get(key, 0)
            row += f"{score:>{col_width - 1}.1f}%"
        row += " ║"
        print(row)
    
    # Overall
    print("  ╟" + "─" * (total_width - 2) + "╢")
    row = f"  ║ {'OVERALL':<24}"
    for m in methods:
        score = results[m].get("overall", 0)
        if score >= 90:
            grade = "A"
        elif score >= 80:
            grade = "B"
        elif score >= 70:
            grade = "C"
        elif score >= 60:
            grade = "D"
        else:
            grade = "F"
        cell = f"{score:.1f}% ({grade})"
        row += f"{cell:>{col_width}}"
    row += " ║"
    print(row)
    print("  ╚" + "═" * (total_width - 2) + "╝")


def _print_metric_explanations():
    """Print brief explanation of what each metric measures."""
    print()
    print("  ┌─ What these metrics mean ──────────────────────────────────────┐")
    print("  │ Distribution     — Do returns follow the same shape?           │")
    print("  │ Moments          — Are mean, std, skew, kurtosis similar?      │")
    print("  │ Autocorrelation  — Does volatility cluster the same way?       │")
    print("  │ Volatility Dyn.  — Are GARCH persistence & reactivity close?   │")
    print("  │ Tail Behavior    — Do extreme moves happen at similar rates?    │")
    print("  │ MMD Score        — Overall statistical distance (kernel-based)  │")
    print("  │                                                                │")
    print("  │ 90%+ = A (excellent)  80%+ = B (good)  70%+ = C (decent)      │")
    print("  │ 60%+ = D (fair)       <60% = F (poor)                          │")
    print("  └────────────────────────────────────────────────────────────────┘")


def save_report(results: dict, run_dir: str, real_csv: str):
    """Save validation results to a text report file."""
    report_dir = os.path.join(config.REPORTS_DIR, os.path.basename(run_dir))
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, "validation_report.txt")
    
    method_labels = {
        "tda": "TDA Topology",
        "transformer": "MarketGPT",
        "diffusion": "DDPM Diffusion",
        "agent": "Agent Simulation",
        "a_tier": "A-Tier Engine",
    }
    
    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║              SYNTHETIC DATA VALIDATION REPORT               ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append(f"  Source data:  {real_csv}")
    lines.append(f"  Run folder:   {run_dir}")
    lines.append("")
    
    methods = list(results.keys())
    
    header = f"  {'Metric':<25}"
    for m in methods:
        header += f"{method_labels.get(m, m):>15}"
    lines.append(header)
    lines.append("  " + "─" * (25 + 15 * len(methods)))
    
    for display_name, key, _ in METRICS:
        row = f"  {display_name:<25}"
        for m in methods:
            score = results[m].get(key, 0)
            row += f"{score:>14.1f}%"
        lines.append(row)
    
    lines.append("  " + "─" * (25 + 15 * len(methods)))
    row = f"  {'OVERALL':<25}"
    for m in methods:
        score = results[m].get("overall", 0)
        if score >= 90: grade = "A"
        elif score >= 80: grade = "B"
        elif score >= 70: grade = "C"
        elif score >= 60: grade = "D"
        else: grade = "F"
        row += f"  {score:>8.1f}% ({grade})"
    lines.append(row)
    
    # Save JSON
    json_path = os.path.join(report_dir, "validation_scores.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\n  📄 Report  → {report_path}")
    print(f"  📄 Scores  → {json_path}")
    
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════



"""
tstr_benchmark.py — Train-on-Synthetic-Test-on-Real benchmark.

Standalone usage:
    python tstr_benchmark.py --real input_data/raw_ohlcv/BTCUSDT_1h.csv --generated generated_data/run_001_.../

Trains a simple direction-prediction model (XGBoost) and compares:
    TRTR  — Train on Real, Test on Real (baseline upper bound)
    TSTR  — Train on Synthetic, Test on Real (quality measure)
    TSTR+R — Train on Synthetic + Real, Test on Real (augmentation test)
"""

import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

import config

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def create_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Create features for a simple direction-prediction model.
    
    Features:
    - Lagged returns (1, 2, 3, 5, 10, 20 periods)
    - Moving average ratios (5, 10, 20, 50 period MAs vs close)
    - Volatility (rolling std of returns over 10, 20 periods)
    - RSI-like momentum (positive returns ratio over 14 periods)
    - Volume change ratio
    
    Target: 1 if next return is positive, 0 otherwise
    """
    closes = df["close"].values.astype(float)
    volumes = df["volume"].values.astype(float)
    
    log_returns = np.diff(np.log(closes))
    
    feat_df = pd.DataFrame({"close": closes[1:], "log_ret": log_returns, "volume": volumes[1:]})
    
    features = pd.DataFrame(index=feat_df.index)
    
    for lag in config.TSTR_FEATURE_LAGS:
        features[f"ret_lag_{lag}"] = feat_df["log_ret"].shift(lag)
    
    for window in config.TSTR_MA_WINDOWS:
        ma = feat_df["close"].rolling(window).mean()
        features[f"ma_ratio_{window}"] = feat_df["close"] / ma - 1.0
    
    for window in [10, 20]:
        features[f"vol_{window}"] = feat_df["log_ret"].rolling(window).std()
    
    for window in [14]:
        positive = (feat_df["log_ret"] > 0).astype(float)
        features[f"rsi_{window}"] = positive.rolling(window).mean()
    
    features["vol_change"] = feat_df["volume"].pct_change()
    
    target = (feat_df["log_ret"].shift(-1) > 0).astype(int)
    
    valid = features.dropna().index
    valid = valid.intersection(target.dropna().index)
    
    X = features.loc[valid].values
    y = target.loc[valid].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X, y


def _train_and_evaluate(X_train, y_train, X_test, y_test) -> dict:
    """Train XGBoost classifier and evaluate."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    directions = 2 * y_pred - 1
    actual_directions = 2 * y_test - 1
    returns = directions * actual_directions
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    
    return {"accuracy": acc, "f1": f1, "sharpe": sharpe}


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_benchmark(real_csv: str, run_dir: str) -> dict:
    """
    Run TSTR benchmark: compare model performance when trained on real vs synthetic data.
    """
    # Load and prepare real data
    real_df = pd.read_csv(real_csv)
    X_real, y_real = create_features(real_df)
    
    split_idx = int(len(X_real) * config.TSTR_TRAIN_RATIO)
    X_real_train, X_real_test = X_real[:split_idx], X_real[split_idx:]
    y_real_train, y_real_test = y_real[:split_idx], y_real[split_idx:]
    
    print(f"\n  📊 Real data: {len(X_real):,} samples")
    print(f"     Train: {len(X_real_train):,}  |  Test: {len(X_real_test):,} (80/20 chronological split)")
    print(f"     Model: XGBoost direction predictor (next candle UP or DOWN)")
    
    results = {}
    
    # ── Baseline: Train on Real, Test on Real (TRTR) ──
    print(f"\n  ⏳ [TRTR] Training on REAL data (baseline)...")
    trtr = _train_and_evaluate(X_real_train, y_real_train, X_real_test, y_real_test)
    results["TRTR (baseline)"] = trtr
    print(f"     → Accuracy: {trtr['accuracy']:.1%}  |  F1: {trtr['f1']:.3f}  |  Sharpe: {trtr['sharpe']:.2f}")
    
    baseline_acc = trtr["accuracy"]
    
    # ── For each synthetic method ──
    method_labels = {
        "diffusion": "Diffusion",
        "transformer": "Transformer",
        "tda": "TDA Engine",
        "agent": "Agent Sim",
        "a_tier": "A-Tier IAAFT"
    }
    
    for method_key in ["tda", "transformer", "diffusion", "agent", "a_tier"]:
        method_dir = os.path.join(run_dir, method_key)
        if not os.path.isdir(method_dir):
            continue
        
        csvs = [f for f in os.listdir(method_dir) if f.endswith(".csv")]
        if not csvs:
            continue
        
        synth_path = os.path.join(method_dir, csvs[0])
        synth_df = pd.read_csv(synth_path)
        
        label = method_labels.get(method_key, method_key)
        
        X_synth, y_synth = create_features(synth_df)
        
        if len(X_synth) < 100:
            print(f"\n  ⚠️  [{label}] Not enough synthetic data, skipping...")
            continue
        
        # ── TSTR: Train on Synthetic, Test on Real ──
        print(f"\n  ⏳ [TSTR {label}] Training on SYNTHETIC data only...")
        tstr = _train_and_evaluate(X_synth, y_synth, X_real_test, y_real_test)
        results[f"TSTR ({label})"] = tstr
        gap = (baseline_acc - tstr["accuracy"]) * 100
        direction = "↓" if gap > 0 else "↑"
        print(f"     → Accuracy: {tstr['accuracy']:.1%}  |  F1: {tstr['f1']:.3f}  |  Sharpe: {tstr['sharpe']:.2f}")
        print(f"     → vs Baseline: {direction} {abs(gap):.1f}% gap")
        
        # ── TSTR+R: Train on Synthetic + Real, Test on Real ──
        print(f"  ⏳ [TSTR+R {label}] Training on SYNTHETIC + REAL combined...")
        X_combined = np.vstack([X_synth, X_real_train])
        y_combined = np.concatenate([y_synth, y_real_train])
        
        shuffle_idx = np.random.permutation(len(X_combined))
        X_combined = X_combined[shuffle_idx]
        y_combined = y_combined[shuffle_idx]
        
        tstr_r = _train_and_evaluate(X_combined, y_combined, X_real_test, y_real_test)
        results[f"TSTR+R ({label})"] = tstr_r
        gap_r = (baseline_acc - tstr_r["accuracy"]) * 100
        direction_r = "↓" if gap_r > 0 else "↑"
        print(f"     → Accuracy: {tstr_r['accuracy']:.1%}  |  F1: {tstr_r['f1']:.3f}  |  Sharpe: {tstr_r['sharpe']:.2f}")
        print(f"     → vs Baseline: {direction_r} {abs(gap_r):.1f}% gap")
    
    # ── Print summary table ──
    _print_benchmark_table(results, baseline_acc)
    _print_explanation()
    
    return results


def _print_benchmark_table(results: dict, baseline_acc: float):
    """Print formatted benchmark comparison table."""
    col_width = 13
    total_width = 32 + col_width * 4
    
    print()
    print("  ╔" + "═" * (total_width - 2) + "╗")
    title = "TSTR BENCHMARK RESULTS"
    print(f"  ║{title:^{total_width - 2}}║")
    print("  ╠" + "═" * (total_width - 2) + "╣")
    
    header = f"  ║ {'Experiment':<29} {'Accuracy':>{col_width}} {'F1':>{col_width}} {'Sharpe':>{col_width}} {'vs Base':>{col_width}} ║"
    print(header)
    print("  ╟" + "─" * (total_width - 2) + "╢")
    
    for name, scores in results.items():
        gap = (baseline_acc - scores["accuracy"]) * 100
        if name.startswith("TRTR"):
            gap_str = "—"
        elif gap > 0:
            gap_str = f"↓ {gap:.1f}%"
        else:
            gap_str = f"↑ {abs(gap):.1f}%"
        
        row = (f"  ║ {name:<29}"
               f" {scores['accuracy']:>{col_width - 1}.1%}"
               f" {scores['f1']:>{col_width - 1}.3f}"
               f" {scores['sharpe']:>{col_width - 1}.2f}"
               f" {gap_str:>{col_width}}"
               f" ║")
        print(row)
    
    print("  ╚" + "═" * (total_width - 2) + "╝")
    
    # ── Verdicts ──
    print()
    for name, scores in results.items():
        if name.startswith("TRTR"):
            continue
        gap = (baseline_acc - scores["accuracy"]) * 100
        if gap < 2:
            verdict = "🟢 Excellent — nearly identical to real data training"
        elif gap < 5:
            verdict = "🟡 Good — minor gap, solid for augmentation"
        elif gap < 10:
            verdict = "🟠 Fair — noticeable gap, adds data diversity only"
        else:
            verdict = "🔴 Poor — synthetic data quality not sufficient"
        
        if gap < 0:
            verdict = "🟢 Excellent — actually outperformed real data training"
        
        print(f"  {name}: {verdict}")


def _print_explanation():
    """Print explanation of what the TSTR numbers mean."""
    print()
    print("  ┌─ How to read these results ────────────────────────────────────┐")
    print("  │                                                                │")
    print("  │ Accuracy = how often the model correctly predicts if the       │")
    print("  │ next candle goes UP or DOWN. 50% = coin flip.                  │")
    print("  │                                                                │")
    print("  │ Markets are nearly efficient, so 52-56% on real data is        │")
    print("  │ normal and actually good. Nobody gets 90% here.                │")
    print("  │                                                                │")
    print("  │ The key number is 'vs Base' — the gap between training on      │")
    print("  │ real vs synthetic data. Smaller gap = better synthetic data.    │")
    print("  │                                                                │")
    print("  │ TRTR   = Train Real, Test Real (best case baseline)            │")
    print("  │ TSTR   = Train Synthetic, Test Real (synthetic quality test)   │")
    print("  │ TSTR+R = Train Synthetic+Real, Test Real (augmentation test)   │")
    print("  │                                                                │")
    print("  │ If TSTR is within ~5% of TRTR, your synthetic data is solid.   │")
    print("  │ If TSTR+R beats TRTR, synthetic data improves your model.      │")
    print("  └────────────────────────────────────────────────────────────────┘")


def save_benchmark_report(results: dict, run_dir: str, real_csv: str):
    """Save benchmark results to file."""
    report_dir = os.path.join(config.REPORTS_DIR, os.path.basename(run_dir))
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, "tstr_results.txt")
    
    baseline_acc = results.get("TRTR (baseline)", {}).get("accuracy", 0.5)
    
    lines = []
    lines.append("╔══════════════════════════════════════════════════════════════╗")
    lines.append("║                  TSTR BENCHMARK RESULTS                     ║")
    lines.append("╚══════════════════════════════════════════════════════════════╝")
    lines.append("")
    lines.append(f"  Source data:  {real_csv}")
    lines.append(f"  Run folder:   {run_dir}")
    lines.append("")
    lines.append(f"  {'Experiment':<30} {'Accuracy':>10} {'F1':>10} {'Sharpe':>10} {'vs Base':>10}")
    lines.append("  " + "─" * 70)
    
    for name, scores in results.items():
        gap = (baseline_acc - scores["accuracy"]) * 100
        if name.startswith("TRTR"):
            gap_str = "—"
        elif gap > 0:
            gap_str = f"↓ {gap:.1f}%"
        else:
            gap_str = f"↑ {abs(gap):.1f}%"
        
        lines.append(f"  {name:<30} {scores['accuracy']:>9.1%} {scores['f1']:>9.3f} {scores['sharpe']:>9.2f} {gap_str:>10}")
    
    # Save JSON
    json_path = os.path.join(report_dir, "tstr_scores.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    report_text = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"\n  📄 Report → {report_path}")
    print(f"  📄 Scores → {json_path}")
    
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════



import os
import pandas as pd
import numpy as np

def detect_order_blocks(df, atr_period=14, impulse_mult=2.0, lookforward=50):
    df = df.copy()
    if 'datetime' not in df.columns and 'open_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['open_time'], unit='ms' if df['open_time'].max() > 1e11 else 's')
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    body = df['close'] - df['open']
    abs_body = np.abs(body)
    
    bullish_obs = []
    bearish_obs = []

    for i in range(1, len(df)-2):
        if pd.isna(atr.iloc[i]):
            continue
            
        if body.iloc[i] > atr.iloc[i] * impulse_mult:
            ob_idx = i - 1
            while ob_idx >= 0 and body.iloc[ob_idx] >= 0:
                ob_idx -= 1
            
            if ob_idx >= 0:
                if df['low'].iloc[i+1] > df['high'].iloc[ob_idx]:
                    bullish_obs.append({
                        'idx': ob_idx,
                        'price': df['high'].iloc[ob_idx],
                        'impulse_size': abs_body.iloc[i]
                    })
                    
        elif body.iloc[i] < -atr.iloc[i] * impulse_mult:
            ob_idx = i - 1
            while ob_idx >= 0 and body.iloc[ob_idx] <= 0:
                ob_idx -= 1
            
            if ob_idx >= 0: 
                if df['high'].iloc[i+1] < df['low'].iloc[ob_idx]:
                    bearish_obs.append({
                        'idx': ob_idx,
                        'price': df['low'].iloc[ob_idx], 
                        'impulse_size': abs_body.iloc[i]
                    })
                    
    mitigated_bullish = 0
    for ob in bullish_obs:
        idx = ob['idx']
        ob_price = ob['price']
        mitigated = False
        for j in range(idx + 3, min(idx + lookforward, len(df))):
            if df['low'].iloc[j] <= ob_price:
                mitigated = True
                break
        if mitigated: mitigated_bullish += 1
            
    mitigated_bearish = 0
    for ob in bearish_obs:
        idx = ob['idx']
        ob_price = ob['price']
        mitigated = False
        for j in range(idx + 3, min(idx + lookforward, len(df))):
            if df['high'].iloc[j] >= ob_price:
                mitigated = True
                break
        if mitigated: mitigated_bearish += 1

    return {
        'total_bullish_ob': len(bullish_obs),
        'total_bearish_ob': len(bearish_obs),
        'bullish_mitigation_rate': mitigated_bullish / len(bullish_obs) if bullish_obs else 0,
        'bearish_mitigation_rate': mitigated_bearish / len(bearish_obs) if bearish_obs else 0,
        'avg_impulse': np.mean([ob['impulse_size'] for ob in bullish_obs + bearish_obs]) if bullish_obs or bearish_obs else 0
    }

def detect_trendlines(df, pivot_len=5, max_lookahead=200, touch_tolerance_pct=0.002):
    prices_high = df['high'].values
    prices_low = df['low'].values
    
    high_pivots = []
    low_pivots = []
    
    for i in range(pivot_len, len(df) - pivot_len):
        if all(prices_high[i] > prices_high[i-pivot_len:i]) and all(prices_high[i] > prices_high[i+1:i+pivot_len+1]):
            high_pivots.append(i)
        if all(prices_low[i] < prices_low[i-pivot_len:i]) and all(prices_low[i] < prices_low[i+1:i+pivot_len+1]):
            low_pivots.append(i)
            
    valid_uptrends = []
    for i in range(len(low_pivots) - 1):
        p1 = low_pivots[i]
        for j in range(i+1, min(i+5, len(low_pivots))):
            p2 = low_pivots[j]
            if p2 - p1 > max_lookahead: pass
            
            slope = (prices_low[p2] - prices_low[p1]) / (p2 - p1)
            if slope <= 0: continue
            
            touches = 2
            broken_idx = -1
            for k in range(p2 + 1, min(p2 + max_lookahead, len(prices_low))):
                expected_price = prices_low[p1] + slope * (k - p1)
                actual_price = prices_low[k]
                
                if df['close'].values[k] < expected_price * (1 - touch_tolerance_pct):
                    broken_idx = k
                    break
                    
                if abs(actual_price - expected_price) / expected_price <= touch_tolerance_pct:
                    touches += 1
            
            if touches >= 3:
                duration = broken_idx - p1 if broken_idx != -1 else max_lookahead
                valid_uptrends.append({
                    'touches': touches,
                    'duration': duration,
                    'slope': slope / prices_low[p1]
                })
                
    valid_downtrends = []
    for i in range(len(high_pivots) - 1):
        p1 = high_pivots[i]
        for j in range(i+1, min(i+5, len(high_pivots))):
            p2 = high_pivots[j]
            if p2 - p1 > max_lookahead: pass
            
            slope = (prices_high[p2] - prices_high[p1]) / (p2 - p1)
            if slope >= 0: continue
            
            touches = 2
            broken_idx = -1
            for k in range(p2 + 1, min(p2 + max_lookahead, len(prices_high))):
                expected_price = prices_high[p1] + slope * (k - p1)
                actual_price = prices_high[k]
                
                if df['close'].values[k] > expected_price * (1 + touch_tolerance_pct):
                    broken_idx = k
                    break
                    
                if abs(actual_price - expected_price) / expected_price <= touch_tolerance_pct:
                    touches += 1
            
            if touches >= 3:
                duration = broken_idx - p1 if broken_idx != -1 else max_lookahead
                valid_downtrends.append({
                    'touches': touches,
                    'duration': duration,
                    'slope': slope / prices_high[p1]
                })

    all_trends = valid_uptrends + valid_downtrends
    
    return {
        'total_uptrends': len(valid_uptrends),
        'total_downtrends': len(valid_downtrends),
        'avg_touches': np.mean([t['touches'] for t in all_trends]) if all_trends else 0,
        'avg_duration': np.mean([t['duration'] for t in all_trends]) if all_trends else 0,
        'avg_abs_slope': np.mean([abs(t['slope']) for t in all_trends]) if all_trends else 0
    }

def process_file(name, path):
    df = pd.read_csv(path)
    if 'open' not in df.columns or 'close' not in df.columns:
        return None
        
    df = df.dropna().reset_index(drop=True)
    n_candles = len(df)
    
    ob_stats = detect_order_blocks(df)
    tl_stats = detect_trendlines(df)
    
    freq_multiplier = 1000 / n_candles if n_candles > 0 else 0
    
    res = {
        'name': name,
        'candles': n_candles,
        
        # OB
        'ob_per_1000': (ob_stats['total_bullish_ob'] + ob_stats['total_bearish_ob']) * freq_multiplier,
        'bull_ob_mitig_pct': ob_stats['bullish_mitigation_rate'] * 100,
        'bear_ob_mitig_pct': ob_stats['bearish_mitigation_rate'] * 100,
        'ob_avg_impulse': ob_stats['avg_impulse'],
        
        # TL
        'tl_per_1000': (tl_stats['total_uptrends'] + tl_stats['total_downtrends']) * freq_multiplier,
        'tl_avg_touches': tl_stats['avg_touches'],
        'tl_avg_duration': tl_stats['avg_duration'],
        'tl_avg_norm_slope': tl_stats['avg_abs_slope'],
    }
    return res

def run_structural_validation(input_csv, run_dir, method, timeframe):
    print("\n  " + "═" * 68)
    print("    STAGE 5/4 (!): STRUCTURAL GEOMETRY COMPARISON (OB/TL)")
    print("  " + "═" * 68)
    print("    -> Detecting Trendlines and OrderBlocks on data and outputs...\n")

    files = {"Real Data": input_csv}
    methods = ["tda", "transformer", "diffusion", "agent"] if method == "all" else [method]
    for m in methods:
        path = os.path.join(run_dir, m, f"BTCUSDT_{timeframe}_synthetic.csv")
        if os.path.exists(path):
            files[f"{m.upper()} Synthetic"] = path

    results = []
    for name, path in files.items():
        try:
            r = process_file(name, path)
            if r: results.append(r)
        except Exception as e:
            print(f"[{name}] Failed: {str(e)}")
            
    if not results:
        print("  ❌ No Structural Data could be compared.")
        return
            
    lookup = {r['name']: r for r in results}
    
    if "Real Data" not in lookup:
        print("  ❌ Real Baseline Data is missing.")
        return
        
    real = lookup["Real Data"]
    
    print("  " + "━" * 68)
    print(f"  🟢 **AUTHENTIC BASELINE DATA** ({real['candles']:,} candles)")
    print("  " + "━" * 68)
    print(f"  [Order Blocks] -> {real['ob_per_1000']:.1f} per 1000 candles")
    print(f"                    (Fill Mitigation: {real['bull_ob_mitig_pct']:.1f}% Bull, {real['bear_ob_mitig_pct']:.1f}% Bear)")
    print(f"  [Trendlines]   -> {real['tl_per_1000']:.1f} per 1000 candles")
    print(f"                    (Average Duration: {real['tl_avg_duration']:.1f} periods | Touches: {real['tl_avg_touches']:.1f})")
    print("\n")

    for m in methods:
        syn_name = f"{m.upper()} Synthetic"
        if syn_name in lookup:
            syn = lookup[syn_name]
            
            diff_ob = 0 if real['ob_per_1000'] == 0 else abs(real['ob_per_1000'] - syn['ob_per_1000']) / real['ob_per_1000'] * 100
            diff_tl = 0 if real['tl_per_1000'] == 0 else abs(real['tl_per_1000'] - syn['tl_per_1000']) / real['tl_per_1000'] * 100
            
            color = "🔴" if diff_ob > 30 else "🟡" if diff_ob > 10 else "🟢"
            
            print("  " + "─" * 68)
            print(f"  {color} **{m.upper()} Output Generation**")
            print("  " + "─" * 68)
            print(f"  Deviation   -> OB Mismatch: {diff_ob:.1f}% | Trendline Mismatch: {diff_tl:.1f}%")
            print(f"  Stats       -> {syn['ob_per_1000']:.1f} OBs/1k (Mitigated: {syn['bull_ob_mitig_pct']:.1f}%) | {syn['tl_per_1000']:.1f} Trendlines/1k")
            print("")
    
    print("  " + "═" * 68)
    print("  ┌─ How to read these results ──────────────────────────────────────┐")
    print("  │                                                                │")
    print("  │ Mismatch (%)  -> Shows how far the AI is from real Bitcoin.    │")
    print("  │                  0% is perfect. >30% means the physics failed. │")
    print("  │                                                                │")
    print("  │ Order Blocks  -> Real Bitcoin prints ~14 per 1000 candles.     │")
    print("  │                  If AI prints 0, it means the synthetic data   │")
    print("  │                  is totally smoothed and purely probabilistic. │")
    print("  │                                                                │")
    print("  │ Mitigation    -> When an Order Block forms, how often does     │")
    print("  │                  price return to fill it? Real BTC is ~59%.    │")
    print("  │                                                                │")
    print("  │ Trendlines    -> Real Bitcoin maintains angled support. If AI  │")
    print("  │                  is way off, it's just a random walk.          │")
    print("  └────────────────────────────────────────────────────────────────┘")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate synthetic market data")
    parser.add_argument("--real", type=str, required=True, help="Path to real OHLCV CSV")
    parser.add_argument("--generated", type=str, required=True, 
                        help="Path to run folder or specific synthetic CSV")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.real):
        print(f"❌ Real data file not found: {args.real}")
        exit(1)
    
    if os.path.isdir(args.generated):
        results = validate_run(args.real, args.generated)
        if results:
            save_report(results, args.generated, args.real)
    else:
        real_df = pd.read_csv(args.real)
        synth_df = pd.read_csv(args.generated)
        real_returns = _get_log_returns(real_df)
        synth_returns = _get_log_returns(synth_df)
        scores = validate_single(real_returns, synth_returns)
        print("\nValidation Scores:")
        for name, key, _ in METRICS:
            print(f"  {name:<25} {scores[key]:>6.1f}%")
        print(f"  {'OVERALL':<25} {scores['overall']:>6.1f}%")
