import json
import re
import time
import requests
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# 加载环境变量
load_dotenv()

# ========= ⚙️ 配置 =========
# 模型卡片目录 (输入)
INPUT_DIR = Path("/Users/hpy/Desktop/LLM4TSGym/model_cards")
# 模型代码目录 (用于提取链接)
MODELS_CODE_DIR = Path("/Users/hpy/Desktop/LLM4TSGym/models")
# 输出目录 (直接覆盖更新)
OUTPUT_DIR = Path("/Users/hpy/Desktop/LLM4TSGym/model_cards")

MODEL_NAME = "gpt-4o"
TIMEOUT = 30

client = OpenAI()

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
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an academic assistant proficient in extracting paper abstracts from webpage text."},
                {"role": "user", "content": prompt},
            ]
        )
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
            current_abstract = data.get("论文摘要", "")
            if current_abstract and len(current_abstract) > 50 and "未找到" not in current_abstract and "失败" not in current_abstract:
                print("  ✅ 已有摘要，跳过")
                continue

            # 1. 寻找链接
            # 仅使用 JSON 文件中已有的链接，不再从代码中提取
            url = data.get("论文链接", "")
            
            # 如果 JSON 里原本没有 "论文链接" 字段，或者为空
            if not url or url in ["无", "代码中未找到链接", "None", "null"]:
                 # 尝试一次从代码提取作为补充，如果用户确认不需要这步，可以注释掉下面这行
                 # 但根据指令 "步骤3不用做，没有就不用提去了"，我们完全跳过从代码提取
                 pass
            
            if url and url not in ["无", "代码中未找到链接", "None", "null"]:
                # 2. 提取摘要
                abstract = fetch_abstract(url)
                if abstract and "未找到摘要" not in abstract:
                    data["论文摘要"] = abstract
                    print("  ✅ 摘要提取成功")
                else:
                    data["论文摘要"] = abstract if abstract else "自动提取失败，请人工核对。"
                    print("  ⚠️ 自动提取失败/未找到")
            else:
                # 显式标记为无
                data["论文链接"] = "无"
                data["论文摘要"] = "无"
                print("  ⚠️ 无链接，跳过")

            # 保存 (直接覆盖原文件)
            json_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            
            # 礼貌性延时
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ 处理文件出错: {e}")

if __name__ == "__main__":
    process_cards()