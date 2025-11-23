"""
本脚本用于对已转换为 JSONL 格式的数据集进行分词
并生成适合 Qwen2-7B-Instruct 模型训练的输入格式。

主要流程：
    1. 加载 JSONL 数据集
    2. 使用 HuggingFace AutoTokenizer 对文本进行编码
    3. 自动拼接为 ChatML 格式，并生成 labels
    4. 保存为 HuggingFace Dataset 格式，供后续训练使用

输入 JSONL 文件
输出 Tokenized 数据集（保存到指定目录，供 Trainer 使用）

处理过程中会跳过格式错误的行，并打印提示
"""

from datasets import Dataset
from transformers import AutoTokenizer
import json
import os

MODEL_PATH = "qwen2-7b-instruct"
JSONL_FILE = "data/train_ready.jsonl"
SAVE_PATH = "data/tokenized"

# 模型 tokenizer 路径
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 设置 pad_token（Qwen2 默认无 pad）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载 JSONL
def load_jsonl(jsonl_file):
    data = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "messages" in obj:
                    data.append({"messages": obj["messages"]})
            except Exception as e:
                print(f"跳过错误行: {line[:50]}... 原因: {e}")
    return data

raw_data = load_jsonl(JSONL_FILE)
dataset = Dataset.from_list(raw_data)

# 编码函数（用 apply_chat_template）
def tokenize_fn(example):
    # 直接调用官方函数，自动拼接 ChatML 格式并生成 labels
    enc = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=True,
        return_dict=True
    )
    return enc

tokenized_dataset = dataset.map(tokenize_fn, batched=False)

# 保存为数据集
os.makedirs(SAVE_PATH, exist_ok=True)
tokenized_dataset.save_to_disk(SAVE_PATH)

print(f"数据集已保存，共 {len(tokenized_dataset)} 条，路径：{SAVE_PATH}")
