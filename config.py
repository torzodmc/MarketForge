"""
config.py — Shared configuration for the Synthetic Market Data Generator.

All paths, defaults, and hyperparameters live here.
"""

import os
from datetime import datetime

# ─── Base Paths ───────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DATA_DIR = os.path.join(PROJECT_ROOT, "input_data")
RAW_OHLCV_DIR = os.path.join(INPUT_DATA_DIR, "raw_ohlcv")
FINGERPRINTS_DIR = os.path.join(INPUT_DATA_DIR, "fingerprints")
GENERATED_DATA_DIR = os.path.join(PROJECT_ROOT, "generated_data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# ─── Binance Data Source ──────────────────────────────────────────────────────
BINANCE_VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines"
BINANCE_VISION_DAILY_BASE = "https://data.binance.vision/data/spot/daily/klines"
BINANCE_API_BASE = "https://api.binance.com/api/v3"

# Binance kline CSV columns (no header in raw files)
BINANCE_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
]

# ─── Supported Timeframes ────────────────────────────────────────────────────
SUPPORTED_TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"]

# Map timeframe strings to approximate minutes for calculations
TIMEFRAME_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1h": 60, "2h": 120, "4h": 240, "6h": 360,
    "8h": 480, "12h": 720, "1d": 1440,
}

# ─── Generator Defaults ──────────────────────────────────────────────────────

# Block Bootstrap
BOOTSTRAP_BLOCK_SIZE = 50    # Number of candles per block
BOOTSTRAP_MIN_BLOCK = 20
BOOTSTRAP_MAX_BLOCK = 100

# GARCH + HMM
HMM_N_STATES = 3             # Number of hidden market states (bull/bear/sideways)
HMM_N_ITER = 200             # EM iterations for HMM fitting
GARCH_P = 1                  # GARCH lag order
GARCH_Q = 1                  # GARCH lag order

# Diffusion (DDPM)
DIFFUSION_WINDOW_SIZE = 64   # Window size for training sequences
DIFFUSION_TIMESTEPS = 200    # Number of diffusion steps
DIFFUSION_CHANNELS = 64      # Hidden channels in denoiser
DIFFUSION_EPOCHS = 300
DIFFUSION_BATCH_SIZE = 128
DIFFUSION_LR = 0.001

# ─── Validator Defaults ──────────────────────────────────────────────────────
VALIDATION_ACF_LAGS = 20     # Number of autocorrelation lags to compare
MMD_KERNEL_BANDWIDTH = 1.0   # RBF kernel bandwidth for MMD

# Weights for overall score (must sum to 1.0)
VALIDATION_WEIGHTS = {
    "distribution": 0.15,
    "moments": 0.15,
    "autocorrelation": 0.20,
    "volatility": 0.20,
    "tail": 0.10,
    "mmd": 0.20,
}

# ─── TSTR Defaults ───────────────────────────────────────────────────────────
TSTR_TRAIN_RATIO = 0.8       # 80% train, 20% test for real data split
TSTR_FEATURE_LAGS = [1, 2, 3, 5, 10, 20]  # Lag features for the benchmark model
TSTR_MA_WINDOWS = [5, 10, 20, 50]          # Moving average windows


# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_run_id() -> str:
    """Generate a unique run ID based on counter and timestamp.
    
    Format: run_NNN_YYYYMMDD_HHMM
    Counter is auto-incremented based on existing folders in generated_data/.
    """
    os.makedirs(GENERATED_DATA_DIR, exist_ok=True)
    
    existing = [d for d in os.listdir(GENERATED_DATA_DIR) 
                if os.path.isdir(os.path.join(GENERATED_DATA_DIR, d)) and d.startswith("run_")]
    
    # Extract counter numbers from existing run folders
    counters = []
    for d in existing:
        try:
            counters.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            pass
    
    next_counter = max(counters, default=0) + 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"run_{next_counter:03d}_{timestamp}"


def candles_per_year(timeframe: str) -> int:
    """Calculate approximate number of candles in one year for a given timeframe.
    
    Crypto markets run 24/7/365, so:
    - 1 year = 365.25 days = 525,960 minutes
    """
    minutes_per_year = 525_960
    tf_minutes = TIMEFRAME_MINUTES.get(timeframe)
    if tf_minutes is None:
        raise ValueError(f"Unknown timeframe: {timeframe}")
    return minutes_per_year // tf_minutes


def ensure_dirs():
    """Create all necessary directories if they don't exist."""
    for d in [INPUT_DATA_DIR, RAW_OHLCV_DIR, FINGERPRINTS_DIR, GENERATED_DATA_DIR, REPORTS_DIR]:
        os.makedirs(d, exist_ok=True)
