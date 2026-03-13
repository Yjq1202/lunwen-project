# LLM-Driven Time Series Model Recommendation System (LLM-TS-Rec)

## 1. 项目概述 (Project Overview)

本项目旨在利用大语言模型（LLM）的语义理解与推理能力，解决时间序列分析中“模型选择困难”的问题。通过构建一套自动化的工作流，系统能够理解数据集的统计特性与语义背景，结合对前沿模型（SOTA Models）代码层面的深度认知，以及最新的学术 Benchmark 结果，自动为特定数据集推荐最适合的模型 Top-3 列表。

与传统的基于规则或单纯统计指标的推荐不同，本系统引入了**元知识（Meta-Knowledge）**层面：
*   **数据集画像**: 不仅看统计指标（FFT, ACF等），还结合 LLM 生成的领域语义描述。
*   **模型知识库**: 深入代码实现细节（Attention, MLP, Decomposition），并联网补充论文摘要。
*   **Benchmark 注入**: 能够从学术论文中自动提取 Benchmark 性能表和先验规则，辅助推理。

## 2. 核心架构 (Core Architecture)

系统由四个核心模块组成，形成从数据理解到最终推荐的完整闭环。

### 模块一：数据集画像构建 (Dataset Profiling)
*   **功能**: 将原始 CSV 时间序列数据转化为 LLM 可读的“数据集卡片 (Dataset Card)”。
*   **流程**:
    1.  **统计特征提取**: 计算周期性 (FFT)、自相关性、线性趋势、偏度、峰度等。
    2.  **语义增强**: LLM 根据统计特征生成自然语言描述（如“具有明显周期的交通流量数据”）。
*   **产物**: `dataset_cards/*.json`

### 模块二：模型知识库构建 (Model Knowledge Base)
*   **功能**: 从模型源代码和学术论文中构建“模型卡片 (Model Card)”。
*   **流程**:
    1.  **代码分析**: 扫描 `models/` 下的 Python 代码，利用 LLM 分析其核心机制（Encoder/Decoder, Attention Type 等）。
    2.  **摘要补充**: 自动从代码中提取论文链接，抓取 arXiv 摘要并补充到卡片中。
*   **产物**: `model_cards/*.json`

### 模块三：Benchmark 知识注入 (Benchmark Injection)
*   **功能**: 从 SOTA 论文中提取性能数据和模型特性规则，作为“专家经验”注入系统。
*   **流程**:
    1.  抓取指定 arXiv 论文 HTML。
    2.  LLM 提取两类信息：`tfb_raw_tables` (原始性能表) 和 `tfb_digest_priors` (模型特性总结)。
*   **产物**: `benchmark_data.json`

### 模块四：智能推荐推理 (Intelligent Recommendation)
*   **功能**: 综合上述信息，模拟专家思维进行推理。
*   **流程**:
    1.  **上下文加载**: 读取 Dataset Cards, Model Cards 和 Benchmark Data。
    2.  **Chain-of-Thought 推理**: LLM 根据数据特性匹配模型机制，并参考 Benchmark 历史表现。
    3.  **输出**: 生成推荐矩阵（Rank 1-3）及详细的决策理由。
*   **产物**: `recommendation_results/`

## 3. 环境安装 (Installation)

### 前置要求
*   Python 3.8+
*   OpenAI API Key (或其他兼容的 LLM API)

### 安装步骤

1.  **克隆项目**
    ```bash
    git clone <repository_url>
    cd LLM4TSGym
    ```

2.  **创建虚拟环境**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # MacOS/Linux
    # .venv\Scripts\activate   # Windows
    ```

3.  **安装依赖**
    ```bash
    pip install openai pandas numpy scipy beautifulsoup4 python-dotenv requests lxml openpyxl
    ```

4.  **配置环境变量**
    在项目根目录创建 `.env` 文件，填入 API Key：
    ```env
    OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    ```

## 4. 使用说明 (Usage)

请按照以下顺序运行脚本以完成完整流程。

> **⚠️ 注意**: 部分脚本（如 `card_build.py`, `data_card_build.py`）中包含硬编码的绝对路径（如 `/Users/hpy/...`）。在运行前，请务必打开这些文件并将 `PROJECT_ROOT` 或相关路径变量修改为您本地的项目路径，或改为使用相对路径。

| 步骤 | 脚本文件 | 功能描述 |
| :--- | :--- | :--- |
| **1** | `data_card_build.py` | **构建数据集画像**。<br>读取 `dataset/` 下的 CSV，生成 JSON 卡片到 `dataset_cards/`。 |
| **2** | `card_build.py` | **构建模型卡片**。<br>分析 `models/` 下的代码，生成初步模型卡片到 `model_cards/`。 |
| **3** | `abstract_parse.py` | **补充论文摘要**。<br>读取模型代码中的链接，抓取摘要更新至 `model_cards/`。 |
| **4** | `benchmark_extractor.py` | **提取 Benchmark**。<br>（可选）从论文抓取最新 Benchmark 数据生成 `benchmark_data.json`。 |
| **5** | `model_selector.py` | **执行推荐**。<br>读取所有数据，生成最终推荐结果到 `recommendation_results/`。 |

**运行示例**:
```bash
# 1. 生成数据卡片
python data_card_build.py

# 2. 生成模型卡片
python card_build.py

# 3. 补充摘要
python abstract_parse.py

# 4. 执行推荐
python model_selector.py
```

## 5. 目录结构 (Directory Structure)

```text
LLM4TSGym/
├── dataset/                # [输入] 原始时间序列 CSV 数据
├── models/                 # [输入] 模型源代码 (.py)
├── layers/                 # [输入] 模型依赖的神经网络层实现
├── dataset_cards/          # [生成] 数据集画像 JSON
├── model_cards/            # [生成] 模型知识库 JSON
├── recommendation_results/ # [生成] 推荐结果 (CSV/JSON)
├── benchmark_data.json     # [生成] 提取的 Benchmark 知识库
├── .env                    # [配置] API Key
├── data_card_build.py      # [脚本] 数据集画像构建
├── card_build.py           # [脚本] 模型卡片构建
├── abstract_parse.py       # [脚本] 论文摘要抓取
├── benchmark_extractor.py  # [脚本] Benchmark 数据提取
├── model_selector.py       # [脚本] 核心推荐引擎
└── README.md               # 项目说明文档
```

## 6. 预期产出 (Expected Output)

运行 `model_selector.py` 后，`recommendation_results/` 目录下将生成：
*   `model_recommendation_ranks.csv`: 汇总所有数据集的 Top-3 推荐模型。
*   `model_recommendation_reasons.csv`: 简要推荐理由。
*   `detailed_reasoning.json`: 包含 LLM 完整思维链（Chain of Thought）的详细日志，解释了为何针对该数据推荐该模型。