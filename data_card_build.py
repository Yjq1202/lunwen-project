import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from scipy.fft import fft, fftfreq
from scipy import stats as scipy_stats
from statsmodels.tsa.stattools import adfuller
import warnings

# Ignore statsmodels warnings
warnings.filterwarnings("ignore")

load_dotenv()

# ========= ⚙️ Configuration =========
DATASET_DIR = Path("dataset") # Relative path
OUTPUT_DIR = Path("dataset_cards") # Relative path
MODEL_NAME = "gpt-4o"
client = OpenAI()

class DatasetProfiler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.numeric_df = None
        self.date_col = None
        self.target_series = None # For primary univariate analysis (usually last column)
        
    def load_data(self):
        try:
            self.df = pd.read_csv(self.file_path)
            # Try to identify date column
            possible_date_cols = ['date', 'time', 'timestamp', 'datetime']
            self.date_col = None
            for col in self.df.columns:
                if col.lower() in possible_date_cols:
                    self.date_col = col
                    break
            
            # Extract numeric columns
            if self.date_col:
                self.numeric_df = self.df.drop(columns=[self.date_col]).select_dtypes(include=[np.number])
            else:
                self.numeric_df = self.df.select_dtypes(include=[np.number])
            
            # If no numeric columns found, try forcing conversion
            if self.numeric_df.empty:
                self.numeric_df = self.df.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
            
            # Identify main target for analysis (last column typically)
            if not self.numeric_df.empty:
                self.target_series = self.numeric_df.iloc[:, -1].dropna()
                
            return True
        except Exception as e:
            print(f"❌ Failed to load data {self.file_path}: {e}")
            return False

    def get_meta_info(self):
        return {
            "dataset_name": self.file_path.stem,
            "file_name": self.file_path.name,
            "task_type": "MTS" if self.numeric_df.shape[1] > 1 else "UTS",
            "num_rows": len(self.df),
            "num_cols": self.numeric_df.shape[1],
            "freq": "Unknown", # To be inferred or set manually later
            "prediction_settings": {
                "horizons": [96, 192, 336, 720],
                "lookback": 96
            }
        }

    def get_basic_stats(self):
        series = self.target_series
        if series is None or len(series) == 0:
            return {}
            
        return {
            "missing_rate": round(float(self.numeric_df.isnull().mean().mean()), 4), # missing rate might be small, keep 4
            "mean": round(float(series.mean()), 2),
            "std": round(float(series.std()), 2),
            "cv": round(float(series.std() / (series.mean() + 1e-6)), 2), # Coefficient of Variation
            "skewness": round(float(scipy_stats.skew(series)), 2),
            "kurtosis": round(float(scipy_stats.kurtosis(series)), 2)
        }

    def get_advanced_stats(self):
        series = self.target_series.values
        if len(series) < 20: # Too short for advanced analysis
            return {}

        stats = {}
        
        # 1. Stationarity (ADF Test)
        try:
            # Truncate for speed if necessary
            adf_result = adfuller(series[:min(5000, len(series))]) 
            stats["adf_p_value"] = round(float(adf_result[1]), 4) # p-value sensitive, keep 4
            stats["is_stationary_adf"] = bool(adf_result[1] < 0.05)
        except:
            stats["adf_p_value"] = 1.0
            stats["is_stationary_adf"] = False

        # 2. Trend Strength (Linear Regression R^2)
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, series)
        stats["trend_linearity_r2"] = round(float(r_value**2), 2)
        stats["trend_slope"] = round(float(slope), 4) # slope can be small

        # 3. Seasonality Strength (FFT)
        try:
            yf = fft(series - np.mean(series))
            xf = fftfreq(len(series), 1)
            powers = np.abs(yf)
            # Filter valid frequencies
            valid_mask = (xf > 1.0/len(series)) & (xf < 0.5)
            if np.sum(valid_mask) > 0:
                top_power_idx = np.argmax(powers[valid_mask])
                top_power = powers[valid_mask][top_power_idx]
                total_power = np.sum(powers[valid_mask])
                stats["seasonality_strength"] = round(float(top_power / (total_power + 1e-9)), 2) # Energy ratio
                stats["main_period"] = round(float(1.0 / xf[valid_mask][top_power_idx]), 2)
            else:
                stats["seasonality_strength"] = 0.0
                stats["main_period"] = 0.0
        except:
            stats["seasonality_strength"] = 0.0

        # 4. Serial Correlation
        try:
            lag1_acf = pd.Series(series).autocorr(lag=1)
            stats["autocorr_lag1"] = round(float(lag1_acf), 2)
        except:
            stats["autocorr_lag1"] = 0.0
            
        return stats

    def generate_tags(self, stats_dict):
        tags = {}
        # Stationarity
        tags["non_stationary"] = stats_dict.get("adf_p_value", 1.0) > 0.05
        # Trend
        tags["strong_trend"] = stats_dict.get("trend_linearity_r2", 0.0) > 0.6 or abs(stats_dict.get("trend_slope", 0)) > 0.01
        # Seasonality (Threshold may need tuning)
        tags["strong_seasonality"] = stats_dict.get("seasonality_strength", 0) > 0.1 
        # Noise/Complexity (Low autocorrelation implies noise)
        tags["high_noise"] = abs(stats_dict.get("autocorr_lag1", 0)) < 0.5 
        # Missing Values
        tags["has_missing_values"] = stats_dict.get("missing_rate", 0) > 0.001
        
        return tags

    def run_profile(self):
        if not self.load_data():
            return None
        
        meta = self.get_meta_info()
        basic_stats = self.get_basic_stats()
        adv_stats = self.get_advanced_stats()
        
        # Merge all stats for tagging
        all_stats = {**basic_stats, **adv_stats}
        tags = self.generate_tags(all_stats)
        
        return {
            "meta": meta,
            "stats": all_stats,
            "tags": tags,
            "csv_head": self.df.head(5).to_csv(index=False)
        }

# ========= LLM Enrichment =========

PROMPT_TEMPLATE = """
You are a senior time series data scientist.
Your goal is to enrich the dataset card with semantic insights based on the provided rigorous statistical profile.

[Input Profile]
{profile_json}

[Input Data Preview]
{csv_head}

[Reasoning Guidelines]
1. Domain Inference: Analyze the filename and column names in the CSV preview to infer the specific domain.
2. Characteristic Analysis: Use the 'stats' and 'tags' fields. 
   - If 'non_stationary' is True, highlight the distribution shift challenge.
   - If 'strong_seasonality' is True, emphasize the cyclic nature.
   - If 'high_noise' is True, mention the difficulty of overfitting or the need for robust models.
3. Model Recommendation: Map characteristics to model architectures.
   - Strong Seasonality -> Frequency-domain (e.g., FEDformer) or Decomposition-based (e.g., Autoformer).
   - Non-stationary/Trend -> Linear baselines (DLinear) or Normalization-based Transformers (PatchTST, Non-stationary Transformer).
   - High Noise/Low Autocorrelation -> Robust models or simple MLPs might perform better than complex Transformers.

[Task]
Generate a JSON object with the following keys:
1. "dataset_name_pretty": (e.g., "Electricity Transformer Temperature (Hourly)")
2. "domain": (e.g., Energy, Traffic, Finance)
3. "description": (A detailed paragraph summarizing the data source, identifying key characteristics like trend/seasonality based on the stats, and pointing out forecasting challenges.)
4. "recommended_model_types": (A list of 3-5 specific model architectures or categories suitable for this data, e.g., ["Decomposition Transformers", "Linear Models", "Frequency-domain Models"], strictly derived from the stats.)
5. "reasoning_for_recommendation": (A brief sentence explaining why these models fit the stats, e.g., "Due to strong seasonality and long sequence length, decomposition-based models are preferred.")

Return ONLY the JSON.
"""

def generate_llm_enrichment(profile):
    print(f"🧠 LLM Enriching: {profile['meta']['file_name']} ...")
    
    # Exclude csv_head from prompt json to save tokens
    profile_for_prompt = {k:v for k,v in profile.items() if k != 'csv_head'}
    
    prompt = PROMPT_TEMPLATE.format(
        profile_json=json.dumps(profile_for_prompt, indent=2),
        csv_head=profile['csv_head']
    )
    
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {}

def main():
    if not DATASET_DIR.exists():
        print(f"❌ Directory not found: {DATASET_DIR}")
        return
        
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for csv_file in sorted(DATASET_DIR.glob("*.csv")):
        # Skip if not csv
        if not csv_file.is_file(): continue
        
        print(f"📊 Profiling {csv_file.name}...")
        profiler = DatasetProfiler(csv_file)
        profile = profiler.run_profile()
        
        if not profile:
            continue
            
        # LLM Enrichment
        llm_info = generate_llm_enrichment(profile)
        
        # Merge Final Card
        final_card = {
            "meta": profile["meta"],
            "stats": profile["stats"],
            "tags": profile["tags"],
            "semantic_info": llm_info
        }
        
        # Save
        out_path = OUTPUT_DIR / f"{csv_file.stem}.json"
        out_path.write_text(json.dumps(final_card, indent=2), encoding='utf-8')
        print(f"✅ Saved card to {out_path}")

if __name__ == "__main__":
    main()