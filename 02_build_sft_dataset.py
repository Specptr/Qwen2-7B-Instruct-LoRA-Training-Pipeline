# 02_build_sft_dataset.py
import os, json
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_PATH = r"qwen2-7b-instruct"
JSONL_FILE  = r"data/train.jsonl"
SAVE_PATH   = r"data/tokenized"
MAX_LENGTH  = 2048

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj.get("messages", [])
            if len(msgs) == 2 and msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant":
                data.append({"messages": msgs})
    return data

raw = load_jsonl(JSONL_FILE)
ds = Dataset.from_list(raw)

def tokenize_and_mask(example):
    msgs = example["messages"]
    user_msg = msgs[0]["content"]
    asst_msg = msgs[1]["content"]

    # full: user + assistant(content)
    full_messages = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": asst_msg},
    ]
    full_ids = tokenizer.apply_chat_template(
        full_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors=None,
    )

    # prefix: user + assistant(role start) 的“生成起点”
    # 关键：add_generation_prompt=True 会在 assistant 开始生成处停止
    prefix_messages = [
        {"role": "user", "content": user_msg},
    ]
    prefix_ids = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )

    # 截断（同时保证 prefix_len 不超过 full_len）
    full_ids = full_ids[:MAX_LENGTH]
    prefix_len = min(len(prefix_ids), len(full_ids))

    input_ids = full_ids
    attention_mask = [1] * len(input_ids)

    labels = input_ids.copy()
    for i in range(prefix_len):
        labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized = ds.map(tokenize_and_mask, remove_columns=ds.column_names)

os.makedirs(SAVE_PATH, exist_ok=True)
tokenized.save_to_disk(SAVE_PATH)
print(f"已保存：{len(tokenized)} 条 -> {SAVE_PATH}")
print("columns:", tokenized.column_names)
