"""
data_ingestion.py — Fetch historical OHLCV data and extract statistical fingerprints.

Standalone usage:
    python data_ingestion.py --pair BTCUSDT --timeframe 1h --start 2020-01-01 --end now

What it does:
    1. Downloads OHLCV from data.binance.vision (bulk monthly CSVs) + REST API fallback
    2. Saves combined CSV to input_data/raw_ohlcv/
    3. Computes statistical fingerprint and saves JSON to input_data/fingerprints/
"""

import os
import io
import json
import zipfile
import argparse
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import acf

import config

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def _download_monthly_klines(pair: str, timeframe: str, year: int, month: int) -> pd.DataFrame | None:
    """Download one month of kline data from data.binance.vision."""
    url = f"{config.BINANCE_VISION_BASE}/{pair}/{timeframe}/{pair}-{timeframe}-{year}-{month:02d}.zip"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=config.BINANCE_KLINE_COLUMNS)
        return df
    except Exception:
        return None


def _download_daily_klines(pair: str, timeframe: str, date_str: str) -> pd.DataFrame | None:
    """Download one day of kline data from data.binance.vision (for recent data)."""
    url = f"{config.BINANCE_VISION_DAILY_BASE}/{pair}/{timeframe}/{pair}-{timeframe}-{date_str}.zip"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f, header=None, names=config.BINANCE_KLINE_COLUMNS)
        return df
    except Exception:
        return None


def _fetch_api_klines(pair: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fallback: fetch klines from Binance REST API with pagination."""
    all_data = []
    current = start_ms
    
    while current < end_ms:
        params = {
            "symbol": pair,
            "interval": timeframe,
            "startTime": current,
            "endTime": end_ms,
            "limit": 1000,
        }
        try:
            resp = requests.get(f"{config.BINANCE_API_BASE}/klines", params=params, timeout=30)
            if resp.status_code != 200:
                break
            data = resp.json()
            if not data:
                break
            all_data.extend(data)
            current = data[-1][6] + 1  # close_time + 1ms
        except Exception:
            break
    
    if not all_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data, columns=config.BINANCE_KLINE_COLUMNS)
    return df


def fetch_ohlcv(pair: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given pair and timeframe.
    
    Uses data.binance.vision for bulk monthly downloads, then fills in
    remaining recent days via daily downloads or API fallback.
    
    Args:
        pair: Trading pair, e.g. 'BTCUSDT'
        timeframe: Candle interval, e.g. '1h'
        start_date: Start date string 'YYYY-MM-DD'
        end_date: End date string 'YYYY-MM-DD' or 'now'
    
    Returns:
        DataFrame with columns: open_time, open, high, low, close, volume, ...
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.now() if end_date.lower() == "now" else datetime.strptime(end_date, "%Y-%m-%d")
    
    print(f"\n📥 Fetching {pair} {timeframe} data from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}...")
    
    all_frames = []
    
    # ── Step 1: Download complete months from data.binance.vision ──
    current_month = datetime(start_dt.year, start_dt.month, 1)
    # Only download months that are fully completed (at least 1 month ago)
    cutoff_month = datetime(end_dt.year, end_dt.month, 1) - timedelta(days=1)
    cutoff_month = datetime(cutoff_month.year, cutoff_month.month, 1)
    
    months_to_fetch = []
    while current_month <= cutoff_month:
        months_to_fetch.append((current_month.year, current_month.month))
        if current_month.month == 12:
            current_month = datetime(current_month.year + 1, 1, 1)
        else:
            current_month = datetime(current_month.year, current_month.month + 1, 1)
    
    if months_to_fetch:
        print(f"   Downloading {len(months_to_fetch)} monthly files from data.binance.vision...")
        for year, month in tqdm(months_to_fetch, desc="   Monthly data", unit="month"):
            df = _download_monthly_klines(pair, timeframe, year, month)
            if df is not None and len(df) > 0:
                all_frames.append(df)
    
    # ── Step 2: Fill remaining days with daily downloads ──
    if cutoff_month < datetime(end_dt.year, end_dt.month, 1):
        fill_start = datetime(cutoff_month.year, cutoff_month.month + 1, 1) if months_to_fetch else start_dt
    else:
        fill_start = start_dt if not months_to_fetch else cutoff_month + timedelta(days=32)
        fill_start = datetime(fill_start.year, fill_start.month, 1)
    
    # Download remaining days
    current_day = fill_start
    daily_count = 0
    days_to_fetch = []
    while current_day <= end_dt - timedelta(days=1):  # Yesterday at latest
        days_to_fetch.append(current_day.strftime("%Y-%m-%d"))
        current_day += timedelta(days=1)
    
    if days_to_fetch:
        print(f"   Downloading {len(days_to_fetch)} daily files for recent data...")
        for date_str in tqdm(days_to_fetch, desc="   Daily data", unit="day"):
            df = _download_daily_klines(pair, timeframe, date_str)
            if df is not None and len(df) > 0:
                all_frames.append(df)
                daily_count += 1
    
    # ── Step 3: API fallback for today's data ──
    if all_frames:
        last_close_time = all_frames[-1]["close_time"].iloc[-1]
        end_ms = int(end_dt.timestamp() * 1000)
        if last_close_time < end_ms:
            print("   Fetching remaining data via Binance API...")
            df = _fetch_api_klines(pair, timeframe, int(last_close_time) + 1, end_ms)
            if len(df) > 0:
                all_frames.append(df)
    else:
        # Full API fallback
        print("   No bulk data found, falling back to Binance API...")
        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)
        df = _fetch_api_klines(pair, timeframe, start_ms, end_ms)
        if len(df) > 0:
            all_frames.append(df)
    
    if not all_frames:
        raise RuntimeError(f"Failed to fetch any data for {pair} {timeframe}")
    
    # ── Combine and clean ──
    combined = pd.concat(all_frames, ignore_index=True)
    
    # Convert numeric columns
    numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume",
                    "trades", "taker_buy_base", "taker_buy_quote"]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    
    combined["open_time"] = pd.to_numeric(combined["open_time"], errors="coerce")
    combined["close_time"] = pd.to_numeric(combined["close_time"], errors="coerce")
    
    # Detect timestamp format (microseconds from 2025+ vs milliseconds before)
    if combined["open_time"].iloc[0] > 1e15:
        combined["open_time"] = combined["open_time"] // 1000
        combined["close_time"] = combined["close_time"] // 1000
    
    # Deduplicate by open_time and sort
    combined = combined.drop_duplicates(subset="open_time").sort_values("open_time").reset_index(drop=True)
    
    # Filter to requested date range
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    combined = combined[(combined["open_time"] >= start_ms) & (combined["open_time"] <= end_ms)]
    combined = combined.reset_index(drop=True)
    
    # Add human-readable datetime column
    combined["datetime"] = pd.to_datetime(combined["open_time"], unit="ms")
    
    print(f"   ✅ Fetched {len(combined):,} candles ({combined['datetime'].iloc[0]} → {combined['datetime'].iloc[-1]})")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL FINGERPRINT
# ═══════════════════════════════════════════════════════════════════════════════

def extract_fingerprint(df: pd.DataFrame) -> dict:
    """
    Compute a statistical fingerprint of the OHLCV data.
    
    Returns a dict with distribution stats, autocorrelations, GARCH estimates,
    and tail analysis — everything needed to characterize the data.
    """
    closes = df["close"].values.astype(float)
    
    # Log returns (avoid log(0))
    log_returns = np.diff(np.log(closes))
    log_returns = log_returns[np.isfinite(log_returns)]
    
    if len(log_returns) < 100:
        raise ValueError("Not enough data points to compute fingerprint (need at least 100)")
    
    fingerprint = {}
    
    # ── Distribution Stats ──
    fingerprint["n_candles"] = len(df)
    fingerprint["n_returns"] = len(log_returns)
    fingerprint["mean_return"] = float(np.mean(log_returns))
    fingerprint["std_return"] = float(np.std(log_returns))
    fingerprint["skewness"] = float(sp_stats.skew(log_returns))
    fingerprint["kurtosis"] = float(sp_stats.kurtosis(log_returns))  # Excess kurtosis
    fingerprint["min_return"] = float(np.min(log_returns))
    fingerprint["max_return"] = float(np.max(log_returns))
    
    # ── Autocorrelation of returns (should be near 0 for efficient markets) ──
    acf_returns = acf(log_returns, nlags=config.VALIDATION_ACF_LAGS, fft=True)
    fingerprint["acf_returns"] = [float(x) for x in acf_returns]
    
    # ── Autocorrelation of absolute returns (captures volatility clustering) ──
    abs_returns = np.abs(log_returns)
    acf_abs = acf(abs_returns, nlags=config.VALIDATION_ACF_LAGS, fft=True)
    fingerprint["acf_abs_returns"] = [float(x) for x in acf_abs]
    
    # ── GARCH(1,1) parameter estimates ──
    try:
        from arch import arch_model
        # Scale returns for numerical stability
        scaled_returns = log_returns * 100
        model = arch_model(scaled_returns, vol="Garch", p=1, q=1, mean="Constant", rescale=False)
        result = model.fit(disp="off", show_warning=False)
        fingerprint["garch_omega"] = float(result.params.get("omega", 0))
        fingerprint["garch_alpha"] = float(result.params.get("alpha[1]", 0))
        fingerprint["garch_beta"] = float(result.params.get("beta[1]", 0))
        fingerprint["garch_persistence"] = fingerprint["garch_alpha"] + fingerprint["garch_beta"]
    except Exception as e:
        fingerprint["garch_omega"] = None
        fingerprint["garch_alpha"] = None
        fingerprint["garch_beta"] = None
        fingerprint["garch_persistence"] = None
        fingerprint["garch_error"] = str(e)
    
    # ── Tail Index (Hill estimator) ──
    try:
        sorted_abs = np.sort(np.abs(log_returns))[::-1]
        k = max(int(len(sorted_abs) * 0.05), 10)  # Top 5% of observations
        top_k = sorted_abs[:k]
        threshold = sorted_abs[k]
        if threshold > 0:
            hill_estimate = 1.0 / np.mean(np.log(top_k / threshold))
            fingerprint["hill_tail_index"] = float(hill_estimate)
        else:
            fingerprint["hill_tail_index"] = None
    except Exception:
        fingerprint["hill_tail_index"] = None
    
    # ── OHLCV-specific stats ──
    fingerprint["mean_volume"] = float(df["volume"].mean())
    fingerprint["mean_range_pct"] = float(((df["high"] - df["low"]) / df["close"]).mean() * 100)
    fingerprint["mean_body_pct"] = float((np.abs(df["close"] - df["open"]) / df["close"]).mean() * 100)
    
    # ── Metadata ──
    fingerprint["start_date"] = str(df["datetime"].iloc[0])
    fingerprint["end_date"] = str(df["datetime"].iloc[-1])
    
    return fingerprint


def print_fingerprint(fp: dict):
    """Pretty-print a fingerprint summary."""
    print("\n📊 Statistical Fingerprint:")
    print(f"   Candles:        {fp['n_candles']:,}")
    print(f"   Mean Return:    {fp['mean_return']:.6f}")
    print(f"   Std Return:     {fp['std_return']:.6f}")
    print(f"   Skewness:       {fp['skewness']:.4f}")
    print(f"   Excess Kurt:    {fp['kurtosis']:.4f}")
    print(f"   Min Return:     {fp['min_return']:.4f}")
    print(f"   Max Return:     {fp['max_return']:.4f}")
    
    if fp.get("garch_alpha") is not None:
        print(f"   GARCH α:        {fp['garch_alpha']:.4f}")
        print(f"   GARCH β:        {fp['garch_beta']:.4f}")
        print(f"   GARCH persist:  {fp['garch_persistence']:.4f}")
    
    if fp.get("hill_tail_index") is not None:
        print(f"   Tail Index:     {fp['hill_tail_index']:.2f}")
    
    print(f"   Avg Range %:    {fp['mean_range_pct']:.3f}%")
    print(f"   Period:         {fp['start_date']} → {fp['end_date']}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_ingestion(pair: str, timeframe: str, start_date: str, end_date: str) -> tuple[pd.DataFrame, dict]:
    """
    Full ingestion pipeline: fetch data, save CSV, compute & save fingerprint.
    
    Caches data — if the same pair+timeframe CSV already exists on disk,
    it will be reused instead of re-downloading (unless force_refresh is set).
    
    Returns:
        Tuple of (DataFrame, fingerprint_dict)
    """
    config.ensure_dirs()
    
    csv_path = os.path.join(config.RAW_OHLCV_DIR, f"{pair}_{timeframe}.csv")
    fp_path = os.path.join(config.FINGERPRINTS_DIR, f"{pair}_{timeframe}_fingerprint.json")
    
    # ── Check cache: reuse existing data if available ──
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\n  📦 Cache hit — loaded {len(df):,} candles from {csv_path}")
        print(f"     (delete this file to force re-download)")
        
        # Also load cached fingerprint if it exists
        if os.path.exists(fp_path):
            with open(fp_path, "r") as f:
                fp = json.load(f)
            print(f"  📦 Cache hit — loaded fingerprint from {fp_path}")
            print_fingerprint(fp)
            return df, fp
        
        # Fingerprint missing, recompute
        print("  ⏳ Recomputing fingerprint...")
        fp = extract_fingerprint(df)
        fp["pair"] = pair
        fp["timeframe"] = timeframe
        with open(fp_path, "w") as f:
            json.dump(fp, f, indent=2)
        print(f"  📄 Fingerprint → {fp_path}")
        print_fingerprint(fp)
        return df, fp
    
    # ── No cache: fetch fresh data ──
    df = fetch_ohlcv(pair, timeframe, start_date, end_date)
    
    df.to_csv(csv_path, index=False)
    print(f"  📄 OHLCV     → {csv_path}")
    
    fp = extract_fingerprint(df)
    fp["pair"] = pair
    fp["timeframe"] = timeframe
    
    with open(fp_path, "w") as f:
        json.dump(fp, f, indent=2)
    print(f"  📄 Fingerprint → {fp_path}")
    
    print_fingerprint(fp)
    return df, fp


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch OHLCV data and extract statistical fingerprints")
    parser.add_argument("--pair", type=str, required=True, help="Trading pair (e.g. BTCUSDT)")
    parser.add_argument("--timeframe", type=str, required=True, help="Candle interval (e.g. 1h)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="now", help="End date (YYYY-MM-DD or 'now')")
    
    args = parser.parse_args()
    
    if args.timeframe not in config.SUPPORTED_TIMEFRAMES:
        print(f"❌ Unsupported timeframe '{args.timeframe}'. Choose from: {config.SUPPORTED_TIMEFRAMES}")
        exit(1)
    
    run_ingestion(args.pair.upper(), args.timeframe, args.start, args.end)
