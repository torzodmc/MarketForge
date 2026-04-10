"""
run.py — Interactive CLI for the Synthetic Market Data Generator.

Usage:
    python run.py

Walks the user through configuration, then runs the full pipeline:
    1. Data Ingestion (fetch + fingerprint)
    2. Generation (selected methods)
    3. Validation (scoring)
    4. TSTR Benchmark (practical test)
"""

import os
import sys
import time

import config


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _banner():
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║          SYNTHETIC MARKET DATA GENERATOR v2.0               ║")
    print("  ║                                                             ║")
    print("  ║  Generate unlimited training data from real market patterns  ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()


class ProgressBar:
    """Simple overall pipeline progress bar."""
    
    STAGES = [
        ("Data Ingestion", 10),
        ("Generation", 50),
        ("Validation", 25),
        ("TSTR Benchmark", 15),
    ]
    
    def __init__(self):
        self.current_stage = 0
        self.total_weight = sum(w for _, w in self.STAGES)
    
    def _completed_weight(self):
        return sum(w for _, w in self.STAGES[:self.current_stage])
    
    def start_stage(self, stage_idx: int):
        self.current_stage = stage_idx
        self._print()
    
    def finish(self):
        self.current_stage = len(self.STAGES)
        self._print()
    
    def _print(self):
        done = self._completed_weight()
        pct = int(done / self.total_weight * 100)
        bar_width = 40
        filled = int(bar_width * done / self.total_weight)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        stage_name = ""
        if self.current_stage < len(self.STAGES):
            stage_name = f" → {self.STAGES[self.current_stage][0]}"
        else:
            stage_name = " → Done!"
            pct = 100
            bar = "█" * bar_width
        
        print(f"\n  ┌─ Pipeline Progress ─────────────────────────────────────────┐")
        print(f"  │  [{bar}] {pct:>3}%{stage_name:<18}│")
        print(f"  └────────────────────────────────────────────────────────────┘")


def _prompt(label: str, description: str, default: str = "", 
            options: list = []) -> str:
    print(f"  [{label}]")
    if description:
        for line in description.split("\n"):
            print(f"    {line}")
    if options:
        for opt in options:
            print(f"      {opt}")
    
    prompt_text = f"    > "
    if default:
        prompt_text = f"    [{default}] > "
    
    while True:
        response = input(prompt_text).strip()
        
        # Explicit fast-skip mechanism for testing
        if (not response or response in ['""', "''", '"', "'"]) and default:
            return default
            
        if response:
            # If user explicitly types '""' but no default exists, it breaks later validations
            # We assume if it falls down here, it's a real response
            return response
            
        print("    ⚠️  Please enter a value.")


def _confirm(summary_lines: list) -> bool:
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║                    RUN CONFIGURATION                        ║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    for line in summary_lines:
        padded = f"  {line}"
        print(f"  ║ {line:<59}║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()
    response = input("  Proceed? [Y/n] > ").strip().lower()
    return response in ("", "y", "yes")


# ═══════════════════════════════════════════════════════════════════════════════
# INPUT COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_inputs() -> dict:
    inputs = {}
    
    print()
    pair = _prompt(
        "1. Data Source Pair",
        "Trading pair to use as source data (Binance).\n"
        "Examples: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT",
        default="BTCUSDT"
    ).upper()
    inputs["pair"] = pair
    
    print()
    timespan = _prompt(
        "2. Input Data Timespan",
        "Date range for source data.\n"
        "Format: [START]-[END]  |  Use 'now' for today.\n"
        "Available: Aug 2017 onwards for most pairs.",
        default="[2020-01-01]-[now]"
    )
    
    timespan = timespan.strip("[] ")
    if "]-[" in timespan:
        parts = timespan.split("]-[")
    elif "]-" in timespan:
        parts = timespan.replace("]", "").replace("[", "").split("-", 1)
    elif "-[" in timespan:
        parts = timespan.replace("]", "").replace("[", "").split("-", 1)
    else:
        clean = timespan.replace("[", "").replace("]", "")
        if clean.lower().endswith("now"):
            parts = [clean[:-4].rstrip("-"), "now"]
        elif len(clean) > 10 and clean[10] == "-":
            parts = [clean[:10], clean[11:]]
        else:
            parts = clean.split("-", 1)
    
    inputs["start_date"] = parts[0].strip("[] -")
    inputs["end_date"] = parts[1].strip("[] -") if len(parts) > 1 else "now"
    
    print()
    timeframe = _prompt(
        "3. Timeframe (Candle Interval)",
        "Applies to both source and generated data.\n"
        "Match this to what your ML model trains on.",
        default="1h",
        options=["1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d"]
    ).lower()
    
    if timeframe not in config.SUPPORTED_TIMEFRAMES:
        print(f"    ⚠️  '{timeframe}' not supported, defaulting to 1h")
        timeframe = "1h"
    inputs["timeframe"] = timeframe
    
    print()
    years_str = _prompt(
        "4. Generated Data Span (years)",
        "How many years of synthetic data to generate?\n"
        "Tip: More = better training, diminishing returns past ~20x.",
        default="10"
    )
    try:
        inputs["years"] = float(years_str)
    except ValueError:
        print("    ⚠️  Invalid number, defaulting to 10 years")
        inputs["years"] = 10.0
    
    inputs["method"] = "a_tier"
    
    return inputs


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(inputs: dict):
    pair = inputs["pair"]
    timeframe = inputs["timeframe"]
    start_date = inputs["start_date"]
    end_date = inputs["end_date"]
    years = inputs["years"]
    method = inputs["method"]
    
    total_start = time.time()
    progress = ProgressBar()
    
    # ═══════════════════════════ STAGE 1 ═══════════════════════════
    progress.start_stage(0)
    print("\n  " + "═" * 58)
    print("    STAGE 1/4: DATA INGESTION")
    print("  " + "═" * 58)
    
    from data_ingestion import run_ingestion
    df, fingerprint = run_ingestion(pair, timeframe, start_date, end_date)
    
    input_csv = os.path.join(config.RAW_OHLCV_DIR, f"{pair}_{timeframe}.csv")
    
    # ═══════════════════════════ STAGE 2 ═══════════════════════════
    progress.start_stage(1)
    print("\n  " + "═" * 58)
    print("    STAGE 2/4: SYNTHETIC DATA GENERATION")
    print("  " + "═" * 58)
    
    from generator import run_generation
    run_id = config.get_run_id()
    gen_results = run_generation(input_csv, years, method, run_id=run_id, timeframe=timeframe)
    
    run_dir = os.path.join(config.GENERATED_DATA_DIR, run_id)
    
    if not any(v is not None for v in gen_results.values()):
        print("\n  ❌ All methods failed. Cannot proceed.")
        return
    
    # ═══════════════════════════ STAGE 3 ═══════════════════════════
    progress.start_stage(2)
    print("\n  " + "═" * 58)
    print("    STAGE 3/4: VALIDATION")
    print("  " + "═" * 58)
    
    from validator import validate_run, save_report
    val_results = validate_run(input_csv, run_dir)
    if val_results:
        save_report(val_results, run_dir, input_csv)
    
    # ═══════════════════════════ STAGE 4 ═══════════════════════════
    progress.start_stage(3)
    print("\n  " + "═" * 58)
    print("    STAGE 4/4: TSTR BENCHMARK")
    print("  " + "═" * 58)
    
    print()
    print("  ┌─ What is TSTR? ────────────────────────────────────────────┐")
    print("  │                                                            │")
    print("  │ TSTR = Train on Synthetic, Test on Real.                   │")
    print("  │                                                            │")
    print("  │ This is the ultimate practical test for synthetic data.     │")
    print("  │ We train an XGBoost model to predict if the next candle    │")
    print("  │ goes UP or DOWN, using 3 different training setups:        │")
    print("  │                                                            │")
    print("  │   TRTR  — Real data only (baseline, best case)             │")
    print("  │   TSTR  — Synthetic data only (quality test)               │")
    print("  │   TSTR+R — Synthetic + Real combined (augmentation test)   │")
    print("  │                                                            │")
    print("  │ If TSTR accuracy is close to TRTR, the synthetic data      │")
    print("  │ captures the same patterns that matter for ML training.    │")
    print("  │ If TSTR+R beats TRTR, synthetic data actively improves     │")
    print("  │ your model beyond what real data alone can do.             │")
    print("  └────────────────────────────────────────────────────────────┘")
    
    from validator import run_benchmark, save_benchmark_report
    bench_results = run_benchmark(input_csv, run_dir)
    if bench_results:
        save_benchmark_report(bench_results, run_dir, input_csv)
    
    # ═══════════════════════════ STAGE 5 ═══════════════════════════
    progress.start_stage(3) # Can reuse the old stage mapping visually 
    
    # Dynamic import and execution
    from validator import run_structural_validation
    run_structural_validation(input_csv, run_dir, method, timeframe)
    
    # ═══════════════════════════ DONE ═══════════════════════════
    progress.finish()
    
    elapsed = time.time() - total_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    
    print()
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print(f"  ║  ✅ COMPLETE!  Total time: {minutes}m {seconds}s" + " " * (33 - len(f"{minutes}m {seconds}s")) + "║")
    print("  ╠══════════════════════════════════════════════════════════════╣")
    print(f"  ║  📁 Data    → {os.path.basename(run_dir):<45}║")
    print(f"  ║  📊 Reports → reports/{os.path.basename(run_dir):<38}║")
    print(f"  ║  📄 Source  → {os.path.basename(input_csv):<45}║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _banner()
    
    print("  " + "═" * 58)
    print("  S-TIER ENGINE INITIALIZED")
    print("  " + "═" * 58)
    print("  Booting up IAAFT Phase-Randomization with Volatility Density Forcing.")
    print()
    
    try:
        inputs = collect_inputs()
    except (KeyboardInterrupt, EOFError):
        print("\n\n  Cancelled.")
        sys.exit(0)
    
    method_display = {
        "a_tier": "S-Tier IAAFT Engine",
    }
    
    summary = [
        f"Source:      {inputs['pair']} (Binance)",
        f"Period:      {inputs['start_date']} to {inputs['end_date']}",
        f"Timeframe:   {inputs['timeframe']}",
        f"Generate:    {inputs['years']} years of synthetic data",
        f"Methods:     {method_display.get(inputs['method'], inputs['method'])}",
    ]
    
    try:
        est_candles = int(inputs["years"] * config.candles_per_year(inputs["timeframe"]))
        summary.append(f"Est. candles: ~{est_candles:,}")
    except Exception:
        pass
    
    if not _confirm(summary):
        print("\n  Cancelled.")
        sys.exit(0)
    
    try:
        run_pipeline(inputs)
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Pipeline interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
