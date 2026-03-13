import pandas as pd
import numpy as np
import random
from pathlib import Path

# Paths
COUNTS_FILE = Path("recommendation_results/model_selection_counts.csv")
BENCHMARK_FILE = Path("reaults_paper/TSGym-vs-SOTA-full_new.xlsx")
OUTPUT_FILE = Path("recommendation_results/llm_vs_sota_results.csv")

# Mappings (Lowercase counts_name -> Standard Benchmark Name)
DATASET_MAP = {
    "electricity": "ECL",
    "exchange_rate": "Exchange",
    "national_illness": "ILI",
    "traffic": "Traffic",
    "weather": "Weather",
    # Keys below match strictly if lowercased, but explicit mapping helps
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
}

MODEL_MAP = {
    "Nonstationary_Transformer": "Nonstationary",
    # Add others if needed
}

def load_benchmark_data(filepath):
    """
    Parses the specific format of the benchmark Excel file.
    Returns:
      benchmark_data: {dataset_name: {pred_len: {model_name: {'mse': val, 'mae': val}}}}
      dataset_order: list of dataset names in the order they appear in the file
    """
    try:
        xl = pd.ExcelFile(filepath)
        df = xl.parse(xl.sheet_names[0], header=None)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return {}, []

    # 1. Parse Model Columns
    model_col_map = {} 
    row0 = df.iloc[0]
    for idx in range(2, len(df.columns), 2):
        model_name = row0.iloc[idx]
        if pd.notna(model_name):
            model_col_map[str(model_name).strip()] = idx

    print(f"Found {len(model_col_map)} models in benchmark: {list(model_col_map.keys())}")

    # 2. Parse Rows for Datasets
    benchmark_data = {}
    dataset_order = []
    current_dataset = None
    
    # Iterate from row 2 (data starts)
    for i in range(2, len(df)):
        row = df.iloc[i]
        ds_val = row.iloc[0]
        
        # Update current dataset if valid (New dataset block starts)
        if pd.notna(ds_val):
            current_dataset = str(ds_val).strip()
            if current_dataset not in dataset_order:
                dataset_order.append(current_dataset)
            
        pred_len = row.iloc[1]
        
        if pd.notna(pred_len) and current_dataset:
            pred_len_str = str(pred_len).strip()
            
            # Initialize dicts if not exists
            if current_dataset not in benchmark_data:
                benchmark_data[current_dataset] = {}
            
            metrics_for_len = {}
            
            for model_name, mse_idx in model_col_map.items():
                mse = row.iloc[mse_idx]
                mae = row.iloc[mse_idx + 1]
                
                metrics_for_len[model_name] = {
                    'mse': mse,
                    'mae': mae
                }
            
            benchmark_data[current_dataset][pred_len_str] = metrics_for_len
            
    return benchmark_data, dataset_order

def get_best_llm_model(counts_series):
    """
    Parses a series of "pos-neg" strings (e.g. "6-4").
    Returns the model name with the highest 'pos' count.
    Break ties randomly.
    """
    best_models = []
    max_pos = -1
    
    for model_name, val_str in counts_series.items():
        if pd.isna(val_str) or not isinstance(val_str, str):
            continue
            
        try:
            parts = val_str.split('-')
            if len(parts) != 2:
                continue
            pos = int(parts[0])
            
            if pos > max_pos:
                max_pos = pos
                best_models = [model_name]
            elif pos == max_pos:
                best_models.append(model_name)
                
        except ValueError:
            continue
            
    if not best_models:
        return None
        
    selected = random.choice(best_models)
    # print(f"  Selected {selected} (Score: {max_pos}, Candidates: {best_models})")
    return selected

def main():
    # 1. Load LLM Counts
    if not COUNTS_FILE.exists():
        print(f"Error: {COUNTS_FILE} not found.")
        return
        
    counts_df = pd.read_csv(COUNTS_FILE, index_col=0)
    print(f"Loaded counts for {len(counts_df.columns)} datasets.")

    # 2. Load Benchmark Data (with Order)
    if not BENCHMARK_FILE.exists():
        print(f"Error: {BENCHMARK_FILE} not found.")
        return
        
    bench_data, dataset_order = load_benchmark_data(BENCHMARK_FILE)
    print(f"Loaded benchmark data for {len(bench_data)} datasets.")
    print(f"Benchmark Dataset Order: {dataset_order}")

    # 3. Process - Iterate by Benchmark Order
    results = []
    
    # We need a reverse map or a way to find the counts_df column from the benchmark name
    # DATASET_MAP maps counts_df_col (lowercase) -> benchmark_name
    # Let's verify mapping for each benchmark dataset
    
    for bench_dataset_name in dataset_order:
        print(f"\nProcessing Benchmark Dataset: {bench_dataset_name}...")
        
        # Find corresponding column in counts_df
        found_counts_col = None
        
        # 1. Try reverse lookup in DATASET_MAP
        for c_col, b_name in DATASET_MAP.items():
            if b_name.lower() == bench_dataset_name.lower():
                # Check if this c_col exists in counts_df
                if c_col in counts_df.columns:
                    found_counts_col = c_col
                # Also check case-insensitive match in counts_df columns
                else:
                    for real_col in counts_df.columns:
                        if real_col.lower() == c_col.lower():
                            found_counts_col = real_col
                            break
                if found_counts_col: break
        
        # 2. If not found via map, try direct case-insensitive match
        if not found_counts_col:
            for real_col in counts_df.columns:
                if real_col.lower() == bench_dataset_name.lower():
                    found_counts_col = real_col
                    break
                    
        if not found_counts_col:
            print(f"  Warning: No matching LLM results found for benchmark dataset '{bench_dataset_name}'")
            continue
            
        print(f"  Mapped to LLM Dataset: {found_counts_col}")

        # Get LLM choice
        llm_model = get_best_llm_model(counts_df[found_counts_col])
        if not llm_model:
            print(f"  No valid model selected for {found_counts_col}")
            continue
        
        print(f"  LLM Selected Model: {llm_model}")

        # Get all pred_lens for this dataset from benchmark
        dataset_bench_data = bench_data[bench_dataset_name]
        
        # Map Model Name to Benchmark Key
        bench_model_name = MODEL_MAP.get(llm_model, llm_model)
        
        # Try to find the correct model name key (case-insensitive check)
        final_model_key = None
        sample_pred_len = list(dataset_bench_data.keys())[0]
        if bench_model_name in dataset_bench_data[sample_pred_len]:
            final_model_key = bench_model_name
        else:
            for bm in dataset_bench_data[sample_pred_len].keys():
                if bm.lower() == bench_model_name.lower():
                    final_model_key = bm
                    break
        
        if not final_model_key:
            print(f"  Warning: Model '{llm_model}' not found in benchmark for {bench_dataset_name}")
            results.append({
                "Dataset": bench_dataset_name, # Use benchmark name for output consistency
                "Pred_Len": "N/A",
                "LLM_Model": llm_model,
                "MSE": None,
                "MAE": None,
                "Note": "Model not found in benchmark"
            })
            continue

        # Iterate over all available pred_lens for this dataset (ORDERED as in input)
        # Note: dicts are ordered in recent python, so `dataset_bench_data` keys should be in read order
        for pred_len, models_metrics in dataset_bench_data.items():
            metrics = models_metrics.get(final_model_key)
            if metrics:
                results.append({
                    "Dataset": bench_dataset_name,
                    "Pred_Len": pred_len,
                    "LLM_Model": llm_model,
                    "MSE": metrics['mse'],
                    "MAE": metrics['mae'],
                    "Note": ""
                })

    # 4. Save
    res_df = pd.DataFrame(results)
    
    res_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Results saved to {OUTPUT_FILE}")
    print(res_df.to_string())

if __name__ == "__main__":
    main()
