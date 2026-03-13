import json
import concurrent.futures
import pandas as pd
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ========= ⚙️ 配置 =========
MODEL_CARDS_DIR = Path("model_cards")
DATASET_CARDS_DIR = Path("dataset_cards")
OUTPUT_DIR = Path("recommendation_results")

# 默认路径定义
DEFAULT_JSON_PATH = Path("benchmark_data.json")
DEFAULT_TEX_PATH = Path("/Users/hpy/Desktop/LLM4TSGym/arXiv-2403.20150v4/main.tex")

MODEL_NAME = "gpt-5.2"
USE_BENCHMARK_CONTEXT = True  # Control whether to add benchmark info to prompt

client = OpenAI()

def load_json_files(directory):
    data = {}
    if not directory.exists():
        print(f"❌ 目录不存在: {directory}")
        return {}
    for f in sorted(directory.glob("*.json")):
        try:
            content = json.loads(f.read_text(encoding="utf-8"))
            # 使用文件名(不含后缀)作为 Key
            key = f.stem 
            data[key] = content
        except Exception as e:
            print(f"⚠️ 读取 {f.name} 失败: {e}")
    return data

def load_benchmark_data(file_path):
    if not file_path.exists():
        print(f"⚠️ Benchmark file not found: {file_path}")
        return None
    
    # Check extension
    if file_path.suffix.lower() == ".tex":
        # Return raw text content directly
        print(f"📄 Loading raw LaTeX content from {file_path.name}...")
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"❌ Failed to read .tex file: {e}")
            return None
    
    # For JSON files
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ Failed to read benchmark file: {e}")
        return None

def get_recommendations(dataset_name, dataset_info, all_models_summary, benchmark_data=None, use_benchmark_context=False, verbose=False):
    """
    核心函数：将单个数据集特征 + 所有模型概览发给 LLM
    """
    if verbose:
        print(f"🤖 正在为数据集 [{dataset_name}] 筛选 Top-3 模型...")

    benchmark_context_str = ""
    if use_benchmark_context and benchmark_data:
        benchmark_context_str = "\n[Benchmark Context]\n"
        
        # Scenario A: Benchmark data is Raw Text (from .tex)
        if isinstance(benchmark_data, str):
            benchmark_context_str += "The following is the raw content of a relevant research paper (LaTeX format). Please use the tables and results within it as ground truth for model performance:\n\n"
            # Limit context if needed, but passing full text as requested by user
            # Truncating slightly to avoid blowing up context window if file is massive, 
            # but aiming to keep relevant result sections.
            benchmark_context_str += benchmark_data[:120000] 
            benchmark_context_str += "\n\n[End of Paper Context]\n"

        # Scenario B: Benchmark data is Structured JSON (from benchmark_data.json)
        elif isinstance(benchmark_data, dict):
            # 1. General Rules
            priors = benchmark_data.get("tfb_digest_priors", {})
            rules = priors.get("characteristic_rules_from_paper_text", [])
            if rules:
                benchmark_context_str += "General Characteristic Rules:\n"
                for r in rules:
                    benchmark_context_str += f"- {r.get('rule', '')}\n"
                benchmark_context_str += "\n"
                
            # 2. Model Signatures
            signatures = priors.get("model_signatures", [])
            if signatures:
                benchmark_context_str += "Model Signatures (Best/Worst Scenarios):\n"
                for sig in signatures:
                    benchmark_context_str += f"- Model: {sig.get('model')}\n"
                    benchmark_context_str += f"  Best When: {', '.join(sig.get('best_when', []))}\n"
                    benchmark_context_str += f"  Worst When: {', '.join(sig.get('worst_when', []))}\n"
                benchmark_context_str += "\n"

            # 3. Specific Dataset Results
            raw_tables = benchmark_data.get("tfb_raw_tables", {})
            mts_results = raw_tables.get("mts_results", {})
            # Try exact match or case-insensitive match
            dataset_res = mts_results.get(dataset_name)
            if not dataset_res:
                # try finding case-insensitive
                for k, v in mts_results.items():
                    if k.lower() == dataset_name.lower():
                        dataset_res = v
                        break
            
            if dataset_res:
                benchmark_context_str += f"Existing Benchmark Results for {dataset_name}:\n"
                benchmark_context_str += json.dumps(dataset_res, indent=2) + "\n"

    prompt = f"""
You are a senior time series data scientist proficient in various deep learning forecasting architectures (Transformer-based, MLP-based, CNN-based, etc.).
You are required to complete a precise model selection task.

[Input Information]
1. **Target Dataset Profile**:
Name: {dataset_name}
{json.dumps(dataset_info, ensure_ascii=False, indent=2)}

2. **Candidate Model Library**:
{json.dumps(all_models_summary, ensure_ascii=False, indent=2)}

3. [Guidance on Benchmark Context]
If [Benchmark Context] information is provided above, please utilize it as follows:
- **General Characteristic Rules**: Use these heuristic rules to validate your analysis of dataset properties.
- **Model Signatures**: Use these to filter candidates. If a model is explicitly listed as performing poorly ("Worst When") on the current data's characteristics, avoid recommending it.
- **Existing Benchmark Results**: These are ground-truth performance metrics for this dataset. **Give high priority** to models that have historically achieved low error (MSE/MAE) on this dataset. Your recommendations should align with these empirical results unless you have a compelling reason to deviate.

[Benchmark Context]：{benchmark_context_str}

[Task Objectives]
Please strictly follow logical reasoning (Chain of Thought) to complete the following two tasks:
1. Select the **Top 3 Most Suitable** models for this dataset from the candidate library (Recommendation List).
2. Select the **Top 3 Least Suitable** models for this dataset from the candidate library (Avoidance List).

**Critical Constraints:**
- The "Recommendation List" and "Avoidance List" must be **mutually exclusive**. A model cannot appear in both lists.
- When selecting models, please **completely ignore** factors related to memory usage, computational complexity, and resource efficiency. Focus **solely on the model's performance effectiveness** (accuracy, robustness, ability to capture patterns) on the specific dataset.

[Reasoning Step Requirements]
When generating JSON, please strictly follow this logical order to fill in the fields:

1. **Step 1: In-depth Dataset Profiling (dataset_analysis)**
   - Analyze the **temporal patterns** of the data (strength of periodicity, significance of trend terms, presence of seasonality).
   - Analyze **task difficulties** (e.g., is the sequence non-stationary? is there distribution shift? how is the noise level?).

2. **Step 2: Candidate Model Initial Screening and Matching (candidate_filtering_logic)**
   - Based on dataset characteristics, judge which architectures are most suitable and which are least suitable.
   - For example: strong periodic data fits frequency domain/decomposition models; non-stationary data with trend fits models with RevIN or decomposition mechanisms.
   - For extremely simple linear trend data, complex Transformers might overfit (should be listed as unsuitable); for long-dependence complex sequences, simple MLPs might underfit (should be listed as unsuitable).

3. **Step 3: Final Ranking and Conclusion**
   - **Recommendation List (recommendations)**: Select the 3 models with the highest comprehensive scores.
   - **Avoidance List (negative_recommendations)**: Select the 3 most unsuitable models (serious mechanism mismatch, high risk of overfitting/underfitting).

[Output Format]
Must be output as a valid JSON object, structured as follows:
{{
  "dataset_analysis": "Briefly describe the analysis of dataset periodicity, trend, noise, and difficulties here... (Must be in English)",
  "candidate_filtering_logic": "Briefly describe the screening logic here... (Must be in English)",
  "recommendations": [
     {{ "rank": 1, "model_name": "Model_Name_Here", "reason": "Reason for recommendation... (Must be in English)" }},
     {{ "rank": 2, "model_name": "Model_Name_Here", "reason": "..." }},
     {{ "rank": 3, "model_name": "Model_Name_Here", "reason": "..." }}
  ],
  "negative_recommendations": [
     {{ "rank": 1, "model_name": "Model_Name_Here", "reason": "Reason for non-recommendation (e.g., mechanism mismatch, high overfitting risk)... (Must be in English)" }},
     {{ "rank": 2, "model_name": "Model_Name_Here", "reason": "..." }},
     {{ "rank": 3, "model_name": "Model_Name_Here", "reason": "..." }}
  ]
}}
    """.strip()
    
    if verbose:
        print("-" * 40)
        print(f"📝 Prompt for {dataset_name}:")
        print(prompt)
        print("-" * 40)

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a rigorous algorithm expert who only outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Output token usage
        if resp.usage:
            print(f"ℹ️ [{dataset_name}] Input Tokens: {resp.usage.prompt_tokens}")

        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="LLM-based Model Selector for Time Series")
    parser.add_argument(
        "--source", 
        type=str, 
        choices=["json", "tex"], 
        default="json",
        help="Source of benchmark data: 'json' for cached benchmark_data.json (faster), 'tex' for extracting directly from arXiv paper (more accurate if json is outdated)."
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NUM_ITERATIONS = 10
    CONCURRENCY_LIMIT = 5

    # Determine Benchmark File
    if args.source == "tex":
        benchmark_file = DEFAULT_TEX_PATH
        print(f"🔵 Mode: Using RAW LaTeX Source ({benchmark_file.name})")
    else:
        benchmark_file = DEFAULT_JSON_PATH
        print(f"🟢 Mode: Using Cached JSON Source ({benchmark_file.name})")

    # 1. 加载所有数据
    print("📂 正在加载卡片...")
    models_data = load_json_files(MODEL_CARDS_DIR)
    datasets_data = load_json_files(DATASET_CARDS_DIR)
    benchmark_data = load_benchmark_data(benchmark_file)

    if not models_data or not datasets_data:
        print("❌ 未找到模型卡片或数据集卡片，请检查目录。")
        return
    
    print(f"✅ 加载了 {len(models_data)} 个模型卡片, {len(datasets_data)} 个数据集卡片。")
    if benchmark_data:
        if args.source == "tex":
             print("✅ 已从 LaTeX 提取并解析 Benchmark 数据。")
        else:
             print("✅ 已加载 Benchmark JSON 数据。")

    # 2. 准备模型信息
    models_summary = models_data

    # 3. 初始化结果容器
    # 我们需要两个数字矩阵来分别记录推荐次数和不推荐次数
    matrix_pos = pd.DataFrame(0, index=list(models_summary.keys()), columns=list(datasets_data.keys()))
    matrix_neg = pd.DataFrame(0, index=list(models_summary.keys()), columns=list(datasets_data.keys()))
    
    # 详细日志列表
    detailed_run_logs = []

    # 4. 准备任务列表
    tasks = []
    for d_name, d_info in datasets_data.items():
        for i in range(NUM_ITERATIONS):
            tasks.append((d_name, d_info, i))
    
    total_tasks = len(tasks)
    print(f"🚀 开始并发执行任务 (并发数: {CONCURRENCY_LIMIT}, 总任务数: {total_tasks}, Benchmark Context: {USE_BENCHMARK_CONTEXT})...")

    # 5. 并发执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY_LIMIT) as executor:
        # 提交所有任务
        # 注意：这里 verbose=False 以避免多线程打印混乱
        future_to_task = {
            executor.submit(get_recommendations, d_name, d_info, models_summary, benchmark_data, USE_BENCHMARK_CONTEXT, False): (d_name, i)
            for d_name, d_info, i in tasks
        }
        
        completed_count = 0
        for future in concurrent.futures.as_completed(future_to_task):
            d_name, i = future_to_task[future]
            completed_count += 1
            
            try:
                result = future.result()
                if result and "recommendations" in result:
                    print(f"[{completed_count}/{total_tasks}] ✅ {d_name} (Iter {i+1}) 完成")
                    
                    # 兼容旧代码逻辑，优先取 dataset_analysis，如果没有则尝试取 analysis
                    analysis = result.get("dataset_analysis", result.get("analysis", ""))
                    filtering_logic = result.get("candidate_filtering_logic", "")
                    
                    full_analysis = f"【数据分析】{analysis}\n【筛选逻辑】{filtering_logic}"
                    
                    # 处理【推荐】列表 (Positive)
                    for item in result.get("recommendations", []):
                        m_name = item.get("model_name", "")
                        rank = item.get("rank", 0)
                        reason = item.get("reason", "")
                        
                        target_key = None
                        if m_name in matrix_pos.index:
                            target_key = m_name
                        else:
                            for idx_name in matrix_pos.index:
                                if idx_name.lower() == m_name.lower():
                                    target_key = idx_name
                                    break
                        
                        if target_key:
                            matrix_pos.loc[target_key, d_name] += 1
                            detailed_run_logs.append({
                                "dataset": d_name,
                                "iteration": i + 1,
                                "type": "Recommended (Pos)",
                                "rank": rank,
                                "model_key": target_key,
                                "model_name_raw": m_name,
                                "reason": reason,
                                "analysis": full_analysis
                            })

                    # 处理【不推荐】列表 (Negative)
                    for item in result.get("negative_recommendations", []):
                        m_name = item.get("model_name", "")
                        rank = item.get("rank", 0)
                        reason = item.get("reason", "")
                        
                        target_key = None
                        if m_name in matrix_neg.index:
                            target_key = m_name
                        else:
                            for idx_name in matrix_neg.index:
                                if idx_name.lower() == m_name.lower():
                                    target_key = idx_name
                                    break
                        
                        if target_key:
                            matrix_neg.loc[target_key, d_name] += 1
                            detailed_run_logs.append({
                                "dataset": d_name,
                                "iteration": i + 1,
                                "type": "Avoid (Neg)",
                                "rank": rank,
                                "model_key": target_key,
                                "model_name_raw": m_name,
                                "reason": reason,
                                "analysis": full_analysis
                            })

                else:
                    print(f"[{completed_count}/{total_tasks}] ⚠️ {d_name} (Iter {i+1}) 无有效结果/失败")
            except Exception as exc:
                print(f"[{completed_count}/{total_tasks}] ❌ {d_name} (Iter {i+1}) 异常: {exc}")

    # 6. 保存结果
    
    # 合并矩阵为 "Pos-Neg" 格式字符串
    matrix_final = pd.DataFrame("", index=matrix_pos.index, columns=matrix_pos.columns)
    for r in matrix_pos.index:
        for c in matrix_pos.columns:
            pos_val = matrix_pos.loc[r, c]
            neg_val = matrix_neg.loc[r, c]
            # 格式： "6-4" (6次推荐，4次不推荐)，如果都是0则留空或显示"0-0"
            if pos_val == 0 and neg_val == 0:
                matrix_final.loc[r, c] = "0-0"
            else:
                matrix_final.loc[r, c] = f"{pos_val}-{neg_val}"

    # (1) 统计矩阵 (CSV)
    counts_path = OUTPUT_DIR / "model_selection_counts.csv"
    matrix_final.to_csv(counts_path)
    print(f"\n✅ 模型选择统计矩阵已保存 (格式: 推荐次数-不推荐次数): {counts_path}")

    # (2) 中间过程详细日志 (CSV)
    details_path = OUTPUT_DIR / "model_selection_run_details.csv"
    pd.DataFrame(detailed_run_logs).to_csv(details_path, index=False)
    print(f"✅ 中间过程详细日志已保存: {details_path}")


if __name__ == "__main__":                                                           
    main()