import json
import os
from pathlib import Path
from dotenv import load_dotenv
from llm_router import chat_completion_with_fallback

# Load environment variables
load_dotenv()

# ========= ⚙️ Configuration =========
# Path to the local LaTeX file
PAPER_PATH = Path("arXiv-2403.20150v4/main.tex")
OUTPUT_FILE = Path("benchmark_data.json")
# 使用环境变量覆盖；为空则走 llm_router 默认模型
MODEL_NAME = os.getenv("BENCHMARK_EXTRACTOR_MODEL", "")


# ========= 🛠️ Utility Functions =========

def read_paper_content(file_path):
    print(f"📄 Reading paper content from: {file_path} ...")
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return None

def extract_benchmark_data(paper_text):
    print("🧠 Calling LLM to extract and aggregate benchmark data (this may take a minute)...")
    
    prompt = f"""
You are an expert research engineer building a "Benchmark Memory Pack" from a research paper's raw LaTeX source.

INPUT:
- Paper Content (LaTeX): See below. (Source: TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods)

GOAL:
Produce a machine-usable benchmark summary that can be injected into a time-series model selection prompt.
The user specifically needs FULL extraction of the benchmark tables to determine SOTA.

CRITICAL INSTRUCTION:
The paper contains two main result tables for Multivariate Time Series Forecasting (MTSF):
1. Table with caption "Multivariate forecasting results I" (Label: Common Multivariate forecasting results)
2. Table with caption "Multivariate forecasting results II" (Label: New Multivariate forecasting results)
You MUST extract EVERY single data point (MAE and MSE) from these tables. Do not summarize or skip rows/columns.

OUTPUT:
Return ONE valid JSON object with two top-level keys:
{{
  "tfb_raw_tables": {{...}},
  "tfb_digest_priors": {{...}}
}}

HARD RULES (must follow):
1) NO hallucination. Only extract what is explicitly in the provided text.
2) Handle LaTeX formatting:
   - Extract numerical values from strings like "\\textbf{{0.224}}", "\\underline{{0.280}}", "\\uuline{{0.161}}".
   - "nan" and "inf" should be extracted as null or string "nan"/"inf" (preferably null for numerical analysis, but string if specific meaning). Let's use string "nan"/"inf" to preserve exact state.
3) Provenance is required for high-level items, but for the massive result tables, you can structure it hierarchically to save tokens, as long as the source table is identified.

STEP-BY-STEP TASK:

A) Benchmark Meta Extraction
- Extract: dataset counts, domains, evaluation strategies, metrics.
- Save under tfb_raw_tables["benchmark_meta"].

B) Dataset Inventory (MTS)
- Extract dataset details (name, domain, etc.) if available in text or tables.
- Save as tfb_raw_tables["mts_datasets"].

C) Method Inventory
- Extract list of methods and their types.
- Save under tfb_raw_tables["methods"].

D) Raw Result Tables (FULL EXTRACTION - PRIORITY)
- Extract the MTS results from the two large LaTeX tables mentioned above.
- Structure:
  tfb_raw_tables["mts_results"] = {{
    "DatasetName": {{
      "Horizon (e.g., 96, 192)": {{
        "MethodName": {{ "mae": 0.123, "mse": 0.456 }}
      }}
    }}
  }}
- Ensure ALL datasets (PEMS04, PEMS-BAY, METR-LA, PEMS08, Traffic, Solar, ETTm1, Weather, ILI, Electricity, ETTh1, Exchange, ETTm2, ETTh2, AQShunyi, AQWan, NN5, Wike2000, Wind, ZafNoo, CzeLan, Covid-19, NASDAQ, NYSE, FRED-MD) are extracted.
- Ensure ALL horizons (24, 36, 48, 60 OR 96, 192, 336, 720) are extracted.
- Ensure ALL methods (PatchTST, Crossformer, etc.) are extracted.

E) Digest / Priors
- Compute "overall_leaderboard_mts" based on the extracted data (who has the most "bold" or best values?).
- Extract "characteristic_rules_from_paper_text" (qualitative rules).
- Extract "model_signatures" (qualitative pros/cons of models).

--- PAPER CONTENT BEGINS ---
{paper_text}
--- PAPER CONTENT ENDS ---
"""

    try:
        resp, _, _ = chat_completion_with_fallback(
            model_override=MODEL_NAME or None,
            messages=[
                {"role": "system", "content": "You are a rigorous research assistant who extracts structured data from scientific papers. You output strict JSON."},
                {"role": "user", "content": prompt}
            ],
            response_json=True,
            extra_kwargs={"temperature": 0.1},  # Low temperature for precision
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM Extraction Failed: {e}")
        return None

def main():
    # 1. Read Content
    if not PAPER_PATH.exists():
        print(f"❌ File not found: {PAPER_PATH}")
        return

    tex_content = read_paper_content(PAPER_PATH)
    if not tex_content:
        return
    
    print(f"📄 Content length: {len(tex_content)} characters")

    # 2. LLM Extraction
    data = extract_benchmark_data(tex_content)
    
    if data:
        # 3. Save
        OUTPUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ Benchmark data extracted and saved to: {OUTPUT_FILE}")
        
        # Validation check
        if "mts_results" in data.get("tfb_raw_tables", {}):
            dataset_count = len(data["tfb_raw_tables"]["mts_results"])
            print(f"📊 Extracted results for {dataset_count} datasets.")
        else:
            print("⚠️ Warning: 'mts_results' key missing in output.")
            
    else:
        print("⚠️ Extraction returned no valid data.")

if __name__ == "__main__":
    main()