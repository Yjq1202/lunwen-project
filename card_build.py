import json
import os
import ast
import sys
from pathlib import Path
from dotenv import load_dotenv
from llm_router import chat_completion_with_fallback

# 加载环境变量 (需要 .env 文件中有 OPENAI_API_KEY)
load_dotenv()

# ========= ⚙️ 基本配置 =========

# 使用的 LLM 模型名称（可通过环境变量覆盖；为空则使用 provider 默认模型）
MODEL_NAME = os.getenv("CARD_BUILD_MODEL", "")
# 项目根目录（默认当前仓库根目录）
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
# 扫描模型代码的目录
MODELS_DIR = Path(os.getenv("MODELS_DIR", PROJECT_ROOT / "models"))
# 输出模型卡片的目录（默认新目录，避免覆盖原有 model_cards）
OUTPUT_DIR = Path(os.getenv("CARD_BUILD_OUTPUT_DIR", PROJECT_ROOT / "model_cards_generated"))

# ========= 📝 提示词模板 (Prompt) =========

MODEL_CARD_PROMPT_TEMPLATE = """
You are a deep learning architect proficient in PyTorch and time series forecasting. Please carefully read the following Python code (including the main model file and its local module dependencies), and deeply analyze the design philosophy and technical details of the model.

[Analysis Objectives]
1. **Fact-Finding**: Identify the actual implemented components (Encoders, Decoders, Attention mechanisms, etc.) from the code.
2. **Inference**: Based on the architecture, infer suitable tasks, strengths, and weaknesses.

[Output Requirements]
Based on the code, output a JSON object structured STRICTLY as follows.
Distinguish clearly between **Evidence** (what is explicitly in the code) and **Inference** (what you deduce).

{{
  "Primary Task Type": "",      // e.g., Long-term Forecasting / Imputation / Classification etc.
  "Model Architecture Category": "",      // e.g., Transformer-based, MLP-based, CNN-based, State Space Model (SSM) etc.

  "Core Mechanism & Structure": "",    // [Deep] Summarize the core architecture using professional terminology.
  "Core Innovations": "",        // [Deep] Point out 1-3 key technical innovations.
  "Key Technical Details": {{       // [Deep] Subdivided technical points
      "Input Embedding": "",      // e.g., Patch Embedding, Temporal Embedding etc.
      "Feature Extraction": "",      // e.g., Full Attention, Sparse Attention, Linear Mixing, 2D Convolution etc.
      "Positional Encoding": "",      // e.g., Absolute, Relative, Learnable, or "No Positional Encoding"
      "Non-stationarity Handling": ""    // e.g., RevIN, Series Decomposition, Normalization etc.
  }},

  "Evidence": {{
    "Direct Observations": [
      {{"claim": "Uses RevIN", "evidence": "Class RevIN found in layers/RevIN.py, used in Model.forward"}}
    ],
    "Unknown / Not in Code": []
  }},

  "Inferred Suitability": {{
    "Strengths": [
      {{"label": "Long Sequence Modeling", "why": "Uses O(L) attention mechanism", "conditions": "When L > 1000", "confidence": "high"}}
    ],
    "Weaknesses": [
      {{"label": "High Memory Usage", "why": "Full attention matrix O(L^2)", "conditions": "On large batches", "confidence": "medium"}}
    ]
  }},

  "Resource Requirements & Complexity": "",  // Roughly explain parameter scale and computational complexity
  "Paper Link": ""           // Full URL if found, else "None"
}}

[Constraints]
1. **Evidence**: Claims in "Evidence" MUST be supported by specific code snippets, file names, or class names found in the provided text.
2. **Inference**: For "Inferred Suitability", you MUST provide 'conditions' (when is this true?) and 'confidence'.
3. Do NOT output the "_source_files" field. This will be added programmatically.

Here is the model code and its dependencies:
```python
{code}
```
""".strip()

REQUIRED_KEYS = [
  "Primary Task Type", "Model Architecture Category",
  "Core Mechanism & Structure", "Core Innovations",
  "Key Technical Details", "Resource Requirements & Complexity", "Paper Link",
  "Evidence", "Inferred Suitability"
]

def build_dual_evidence_fields(card: dict, source_files: list) -> dict:
    """
    将已有的代码分析结果映射为“文献+代码双证据”结构的基础骨架。
    说明：
    - card_build 阶段只负责代码证据；论文摘要在 abstract_parse.py 阶段补全。
    - 保留旧字段以兼容已有流程，同时新增结构化字段便于后续一致性审计。
    """
    paper_link = card.get("Paper Link", "None") or "None"

    # 从原有 Evidence 字段复用直接观察
    evidence_block = card.get("Evidence", {}) if isinstance(card.get("Evidence"), dict) else {}
    direct_obs = evidence_block.get("Direct Observations", [])
    if not isinstance(direct_obs, list):
        direct_obs = []

    card["Paper Abstract"] = card.get("Paper Abstract", "None")
    card["paper_evidence"] = {
        "paper_link": paper_link,
        "abstract": card.get("Paper Abstract", "None"),
        "paper_claims": card.get("paper_evidence", {}).get("paper_claims", [])
        if isinstance(card.get("paper_evidence"), dict) else []
    }

    card["code_evidence"] = {
        "source_files": source_files,
        "direct_observations": direct_obs
    }

    # 一致性字段先给空骨架，待摘要补全过程填充
    card["consistency_check"] = card.get("consistency_check", []) if isinstance(card.get("consistency_check"), list) else []
    card["final_claims"] = card.get("final_claims", {
        "supported_by_both": [],
        "code_only": [],
        "paper_only": []
    })

    return card

def module_to_path(module_str):
    return PROJECT_ROOT.joinpath(*module_str.split(".")).with_suffix(".py")

def resolve_local_imports(file_path: Path):
    """
    解析给定 Python 文件中的本地导入依赖。
    支持相对导入和常见包 (layers, models, utils, data_provider)。
    """
    local_deps = {}
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"⚠️ AST parse failed {file_path}: {e}")
        return local_deps

    for node in ast.walk(tree):
        # 处理 from ... import ...
        if isinstance(node, ast.ImportFrom):
            # 处理相对导入：from .a.b import c
            if node.level and node.module:
                # 以当前文件目录为基准回退 level 层
                base = file_path.parent
                try:
                    for _ in range(node.level - 1):
                        base = base.parent
                    rel_path = base.joinpath(*node.module.split(".")).with_suffix(".py")
                    if rel_path.exists():
                        local_deps[f"(relative){node.module}"] = rel_path
                except Exception:
                    pass # 路径越界等忽略
                continue
            
            # 处理绝对导入
            module = node.module or ""
            if module.startswith(("layers", "models", "utils", "data_provider")):
                dep_path = module_to_path(module)
                if dep_path.exists():
                    local_deps[module] = dep_path

        # 处理 import ...
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name.startswith(("layers", "models", "utils", "data_provider")):
                    dep_path = module_to_path(name)
                    if dep_path.exists():
                        local_deps[name] = dep_path

    return local_deps

def read_code_with_dependencies(file_path: Path, max_files=12, max_depth=2):
    """
    递归读取主文件及其依赖，限制深度和文件数量。
    """
    visited = set()
    ordered_files = []

    def add_file(p: Path, depth: int):
        if p in visited or depth > max_depth or len(ordered_files) >= max_files:
            return
        visited.add(p)
        ordered_files.append(p)

        deps = resolve_local_imports(p)
        for _, dp in deps.items():
            add_file(dp, depth + 1)

    add_file(file_path, depth=0)

    combined = []
    source_files = []
    for p in ordered_files:
        try:
            code = p.read_text(encoding="utf-8")
        except Exception as e:
            print(f"⚠️ read failed {p}: {e}")
            continue
        
        rel_path = p.relative_to(PROJECT_ROOT) if p.is_relative_to(PROJECT_ROOT) else p.name
        combined.append(f"# ========= FILE: {rel_path} =========\n{code}\n")
        source_files.append(str(rel_path))

    return "\n\n".join(combined), source_files

def normalize_card(card: dict) -> dict:
    """
    校验并补全必要字段，防止解析失败。
    """
    for k in REQUIRED_KEYS:
        card.setdefault(k, "unknown")
    
    # 确保 Key Technical Details 是字典
    if "Key Technical Details" not in card or not isinstance(card["Key Technical Details"], dict):
        card["Key Technical Details"] = {
            "Input Embedding": "unknown",
            "Feature Extraction": "unknown",
            "Positional Encoding": "unknown",
            "Non-stationarity Handling": "unknown"
        }
    
    # 确保 Evidence 结构
    if "Evidence" not in card or not isinstance(card["Evidence"], dict):
        card["Evidence"] = {"Direct Observations": [], "Unknown / Not in Code": []}

    # 确保 Inferred Suitability 结构
    if "Inferred Suitability" not in card or not isinstance(card["Inferred Suitability"], dict):
        card["Inferred Suitability"] = {"Strengths": [], "Weaknesses": []}

    return card

def generate_model_card(py_path: Path, out_dir: Path):
    if py_path.name == "__init__.py":
        return

    print(f"🔄 正在处理: {py_path.name} ...")
    
    # 使用增强的递归读取函数
    code_content, source_files = read_code_with_dependencies(py_path, max_files=20, max_depth=5)
    
    if not code_content:
        print(f"❌ 读取文件内容为空或失败 {py_path}")
        return

    prompt = MODEL_CARD_PROMPT_TEMPLATE.format(code=code_content)

    try:
        resp, provider, used_model = chat_completion_with_fallback(
            model_override=MODEL_NAME or None,
            response_json=True,
            messages=[
                {"role": "system", "content": "You are a deep learning model analysis assistant who strictly follows instructions."},
                {"role": "user", "content": prompt},
            ],
        )
        print(f"ℹ️ 使用提供方: {provider}, 模型: {used_model}")

        content = resp.choices[0].message.content
        card = json.loads(content)

        # 规范化校验
        card = normalize_card(card)

        # 强制使用文件名作为模型名称，并放在第一位
        final_card = {"model_name": py_path.stem}
        final_card.update(card)
        
        # 记录所有读取的源文件
        final_card["_source_files"] = source_files

        # 新增“文献+代码双证据”骨架字段
        final_card = build_dual_evidence_fields(final_card, source_files)
        
        card = final_card

        # 保存为单独的 JSON 文件
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{py_path.stem}.json"
        out_path.write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
        
        print(f"✅ 生成模型卡: {out_path.name} (包含了 {len(source_files)} 个源文件)")

    except Exception as e:
        print(f"❌ 处理失败 {py_path.name}: {e}")

if __name__ == "__main__":
    if not MODELS_DIR.exists():
        print(f"❌ 目录不存在: {MODELS_DIR}")
    else:
        print(f"🚀 开始扫描目录: {MODELS_DIR}")
        
        for py_path in sorted(MODELS_DIR.glob("*.py")):
            generate_model_card(py_path, OUTPUT_DIR)
            
        print(f"🏁 全部完成，结果已保存至: {OUTPUT_DIR}")