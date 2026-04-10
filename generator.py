import os
import pandas as pd
import config
from a_tier_engine import ATierEngine

def run_generation(input_csv, years, method, run_id, timeframe):
    df = pd.read_csv(input_csv)
    est_candles = int(years * config.candles_per_year(timeframe))
    
    results = {}
    
    # S-Tier Engine (Chunked IAAFT) is actively forced
    engine = ATierEngine()
    engine.train(df)
    df_sync = engine.generate(est_candles)
    out_dir = os.path.join(config.GENERATED_DATA_DIR, run_id, "a_tier")
    os.makedirs(out_dir, exist_ok=True)
    df_sync.to_csv(os.path.join(out_dir, f"BTCUSDT_{timeframe}_synthetic.csv"), index=False)
    results["a_tier"] = True

    return results
