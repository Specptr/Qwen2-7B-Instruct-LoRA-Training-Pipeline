"""
将原始对话文本转换为 Chat 格式 JSONL。
输入格式示例：
    你：你的生日是哪一天
    她：11月13日
输出格式示例：
    {"messages": [{"role": "user", "content": "你的生日是哪一天"},{"role": "assistant", "content": "11月13日"}]}
"""
import re
import json

# 输入文件
INPUT_FILE = "data/raw_text.txt"
# 输出文件
OUTPUT_FILE = "data/train_ready.jsonl"



pattern = re.compile(r"(\d+)\s*你：(.*?)\s*她：(.*)")

rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

for match in pattern.finditer(text):
    user_msg = match.group(2).strip()
    assistant_msg = match.group(3).strip()

    msg_block = {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }

    rows.append(msg_block)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"转换完成，共 {len(rows)} 段对话，已输出至 {OUTPUT_FILE}")
