import json
import os
import ast
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量 (需要 .env 文件中有 OPENAI_API_KEY)
load_dotenv()

# ========= ⚙️ 基本配置 =========

# 初始化 OpenAI 客户端
client = OpenAI()

# 使用的 LLM 模型名称
MODEL_NAME = "gpt-5.2"
# 项目根目录
PROJECT_ROOT = Path("/Users/hpy/Desktop/LLM4TSGym")
# 扫描模型代码的目录
MODELS_DIR = PROJECT_ROOT / "models"
# 输出模型卡片的目录
OUTPUT_DIR = PROJECT_ROOT / "model_cards"

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
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},  # 开启 JSON 模式
            messages=[
                {"role": "system", "content": "You are a deep learning model analysis assistant who strictly follows instructions."},
                {"role": "user", "content": prompt},
            ],
        )

        content = resp.choices[0].message.content
        card = json.loads(content)

        # 规范化校验
        card = normalize_card(card)

        # 强制使用文件名作为模型名称，并放在第一位
        final_card = {"model_name": py_path.stem}
        final_card.update(card)
        
        # 记录所有读取的源文件
        final_card["_source_files"] = source_files
        
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