import json
import os
import re
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from llm_router import chat_completion_with_fallback

# 加载环境变量
load_dotenv()

# ========= ⚙️ 配置 =========
# 项目根目录（默认当前仓库根目录）
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parent))
# 模型卡片目录 (输入，默认读取新生成目录)
INPUT_DIR = Path(os.getenv("ABSTRACT_PARSE_INPUT_DIR", PROJECT_ROOT / "model_cards_generated"))
# 模型代码目录 (用于提取链接)
MODELS_CODE_DIR = Path(os.getenv("MODELS_CODE_DIR", PROJECT_ROOT / "models"))
# 输出目录（默认写入新目录，避免覆盖原有 model_cards）
OUTPUT_DIR = Path(os.getenv("ABSTRACT_PARSE_OUTPUT_DIR", INPUT_DIR))

MODEL_NAME = os.getenv("ABSTRACT_PARSE_MODEL", "")
TIMEOUT = 30


# ========= 🛠️ 工具函数 =========

def normalize_whitespace(s: str) -> str:
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def html_to_text(html: str) -> str:
    # 优先尝试 lxml，如果没有安装则回退到 html.parser
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
        
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return normalize_whitespace(text)

def extract_link_from_code(py_filename: str) -> str:
    """从 Python 源代码中提取论文链接 (通常在 docstring 中)"""
    if not py_filename:
        return ""
    
    code_path = MODELS_CODE_DIR / py_filename
    if not code_path.exists():
        return ""
        
    try:
        content = code_path.read_text(encoding="utf-8")
        # 常见模式: Paper link: https://...
        match = re.search(r"Paper link:\s*(https?://[^\s\"']+)", content, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # 备用模式: 直接找 arxiv 链接
        match = re.search(r"(https?://arxiv\.org/[^\s\"']+)", content, re.IGNORECASE)
        if match:
            return match.group(1)
            
        # 备用模式: openreview
        match = re.search(r"(https?://openreview\.net/[^\s\"']+)", content, re.IGNORECASE)
        if match: 
            return match.group(1)
    except Exception as e:
        # 捕获读取文件或正则匹配中的任何错误
        print(f"  ❌ 从代码中提取链接失败: {e}")
        pass
    return ""

def get_card_link(data: dict) -> str:
    """兼容新旧字段，优先读英文 schema。"""
    url = data.get("Paper Link", "")
    if not url:
        url = data.get("论文链接", "")
    return url


def set_card_link(data: dict, url: str):
    """统一写入英文 schema，同时回写中文字段以兼容旧流程。"""
    data["Paper Link"] = url
    data["论文链接"] = url


def get_card_abstract(data: dict) -> str:
    """兼容新旧字段，优先读英文 schema。"""
    abstract = data.get("Paper Abstract", "")
    if not abstract:
        abstract = data.get("论文摘要", "")
    return abstract


def set_card_abstract(data: dict, abstract: str):
    """统一写入英文 schema，同时回写中文字段以兼容旧流程。"""
    data["Paper Abstract"] = abstract
    data["论文摘要"] = abstract


def extract_paper_claims(abstract: str):
    """
    从摘要中抽取论文主张。
    失败时返回空列表，避免中断全流程。
    """
    if not abstract or abstract in ["无", "Abstract not found"]:
        return []

    prompt = f"""
You are an academic information extraction assistant.
Extract 3-6 concise technical claims from the paper abstract below.
Return strict JSON with this shape:
{{"paper_claims": [{{"claim": "...", "source": "abstract"}}]}}

Abstract:
{abstract[:6000]}
    """.strip()

    try:
        resp, provider, used_model = chat_completion_with_fallback(
            model_override=MODEL_NAME or None,
            response_json=True,
            messages=[
                {"role": "system", "content": "You extract factual paper claims and only output JSON."},
                {"role": "user", "content": prompt},
            ]
        )
        print(f"  ℹ️ 论文主张提取使用: {provider}/{used_model}")
        parsed = json.loads(resp.choices[0].message.content)
        claims = parsed.get("paper_claims", [])
        return claims if isinstance(claims, list) else []
    except Exception as e:
        print(f"  ⚠️ 论文主张提取失败，已跳过: {e}")
        return []


def build_consistency_check(paper_claims, code_observations):
    """
    用 LLM 对论文主张与代码观察做一致性审计。
    返回标准结构，失败时返回空列表。
    """
    if not paper_claims:
        return []

    prompt = f"""
You are a strict reviewer.
Given paper claims and code observations, output a consistency audit.
Return strict JSON with this schema:
{{
  "consistency_check": [
    {{"item": "...", "paper": "supported/not-mentioned", "code": "implemented/not-implemented", "verdict": "consistent/partially-consistent/inconsistent"}}
  ],
  "final_claims": {{
    "supported_by_both": ["..."],
    "code_only": ["..."],
    "paper_only": ["..."]
  }}
}}

Paper Claims:
{json.dumps(paper_claims, ensure_ascii=False, indent=2)}

Code Observations:
{json.dumps(code_observations, ensure_ascii=False, indent=2)}
    """.strip()

    try:
        resp, provider, used_model = chat_completion_with_fallback(
            model_override=MODEL_NAME or None,
            response_json=True,
            messages=[
                {"role": "system", "content": "You perform consistency checks and only output JSON."},
                {"role": "user", "content": prompt},
            ]
        )
        print(f"  ℹ️ 一致性审计使用: {provider}/{used_model}")
        parsed = json.loads(resp.choices[0].message.content)
        checks = parsed.get("consistency_check", [])
        final_claims = parsed.get("final_claims", {
            "supported_by_both": [],
            "code_only": [],
            "paper_only": []
        })
        if not isinstance(checks, list):
            checks = []
        if not isinstance(final_claims, dict):
            final_claims = {
                "supported_by_both": [],
                "code_only": [],
                "paper_only": []
            }
        return checks, final_claims
    except Exception as e:
        print(f"  ⚠️ 一致性审计失败，已跳过: {e}")
        return [], {
            "supported_by_both": [],
            "code_only": [],
            "paper_only": []
        }

def get_abstract_via_llm(text_content: str, source_url: str) -> str:
    """将抓取到的文本喂给 LLM 提取摘要"""
    prompt = f"""
    Below is text content scraped from a webpage about a research paper (Source: {source_url}).
    Please extract the **Abstract** section of the paper from it.

    Requirements:
    1. Only output the body content of the abstract.
    2. If there is an English abstract, prioritize outputting English; if there is no English but only Chinese, output Chinese.
    3. If no abstract can be found in the text at all, please output "Abstract not found" directly.
    4. Do not include the "Abstract" title itself, and do not include the author list.

    Webpage Text Snippet:
    {text_content[:15000]} 
    """ # Truncate to prevent token overflow

    try:
        resp, provider, used_model = chat_completion_with_fallback(
            model_override=MODEL_NAME or None,
            response_json=False,
            messages=[
                {"role": "system", "content": "You are an academic assistant proficient in extracting paper abstracts from webpage text."},
                {"role": "user", "content": prompt},
            ]
        )
        print(f"  ℹ️ 摘要提取使用: {provider}/{used_model}")
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ❌ LLM 提取失败: {e}")
        return ""

def fetch_abstract(url: str) -> str:
    """尝试从 URL 抓取并提取摘要"""
    if not url:
        return ""

    print(f"  🌐 正在抓取: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    
    try:
        # 如果是 Arxiv PDF，尝试转换为 abs 页面 (通常更有利于提取文本)
        if "arxiv.org/pdf/" in url:
            url = url.replace("/pdf/", "/abs/").replace(".pdf", "")
            print(f"  🔀 转换为 abs 链接: {url}")

        resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        resp.raise_for_status()
        
        # 简单的内容类型检查
        content_type = resp.headers.get("Content-Type", "").lower()
        
        if "application/pdf" in content_type:
            return "PDF文件，暂不支持直接解析，请手动补充。"
        
        text = html_to_text(resp.text)
        if len(text) < 500:
            return "网页内容过少，可能是动态渲染页面。"
            
        return get_abstract_via_llm(text, url)

    except Exception as e:
        print(f"  ❌ 抓取失败: {e}")
        return ""

def process_cards():
    if not INPUT_DIR.exists():
        print(f"❌ 输入目录不存在: {INPUT_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取所有 json 文件
    json_files = list(INPUT_DIR.glob("*.json"))
    total_files = len(json_files)
    
    print(f"🚀 开始处理 {total_files} 个文件...")

    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{total_files}] 📄 {json_file.name}")
        
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            
            # 检查是否已有有效摘要
            current_abstract = get_card_abstract(data)
            if current_abstract and len(current_abstract) > 50 and "未找到" not in current_abstract and "失败" not in current_abstract:
                print("  ✅ 已有摘要，跳过")
                abstract = current_abstract
            else:
                abstract = ""

            # 1. 寻找链接
            # 优先读取卡片链接，缺失时回退到模型源码注释中的 Paper link
            url = get_card_link(data)
            
            # 如果 JSON 里原本没有 "论文链接" 字段，或者为空
            if not url or url in ["无", "代码中未找到链接", "None", "null"]:
                 py_name = f"{json_file.stem}.py"
                 url = extract_link_from_code(py_name)
            
            if not abstract and url and url not in ["无", "代码中未找到链接", "None", "null"]:
                # 2. 提取摘要
                abstract = fetch_abstract(url)
                if abstract and "未找到摘要" not in abstract:
                    set_card_abstract(data, abstract)
                    print("  ✅ 摘要提取成功")
                else:
                    set_card_abstract(data, abstract if abstract else "自动提取失败，请人工核对。")
                    print("  ⚠️ 自动提取失败/未找到")
            elif not abstract:
                # 显式标记为无
                set_card_link(data, "无")
                set_card_abstract(data, "无")
                print("  ⚠️ 无链接，跳过")

            # 无论摘要是否新抓取，都统一填充/同步关键字段
            final_link = url if url else get_card_link(data) or "无"
            final_abstract = get_card_abstract(data) or abstract or "无"
            set_card_link(data, final_link)
            set_card_abstract(data, final_abstract)

            # 构建双证据字段
            paper_claims = extract_paper_claims(final_abstract)
            code_obs = []
            if isinstance(data.get("Evidence"), dict):
                code_obs = data["Evidence"].get("Direct Observations", [])
            if not isinstance(code_obs, list):
                code_obs = []

            checks, final_claims = build_consistency_check(paper_claims, code_obs)

            data["paper_evidence"] = {
                "paper_link": final_link,
                "abstract": final_abstract,
                "paper_claims": paper_claims
            }
            data["code_evidence"] = {
                "source_files": data.get("_source_files", []),
                "direct_observations": code_obs
            }
            data["consistency_check"] = checks
            data["final_claims"] = final_claims

            # 保存到新目录（不覆盖输入目录）
            out_path = OUTPUT_DIR / json_file.name
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✅ 已写入: {out_path}")
            
            # 礼貌性延时
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ 处理文件出错: {e}")

if __name__ == "__main__":
    process_cards()