# 01_text_to_jsonl.py
import json

INPUT_FILE = r"data/raw.txt"
OUTPUT_FILE = r"data/train.jsonl"

def parse_blocks(lines):
    """
    期望格式：
    101
    我：...
    她：...
    允许内容里有冒号、空格、标点；不允许跨多行（你当前语料是单行，保持一致最稳）
    """
    i = 0
    n = len(lines)
    blocks = []
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        # 编号行
        if line.isdigit():
            idx = line
            if i + 2 >= n:
                break
            u = lines[i + 1].rstrip("\n")
            a = lines[i + 2].rstrip("\n")
            if not u.strip().startswith("我：") or not a.strip().startswith("她："):
                # 不是严格块，跳过
                i += 1
                continue
            user = u.strip()[2:].strip()
            assistant = a.strip()[2:].strip()
            if user and assistant:
                blocks.append((idx, user, assistant))
            i += 3
            continue

        i += 1

    return blocks

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

blocks = parse_blocks(lines)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for idx, user, assistant in blocks:
        obj = {
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant}
            ]
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"完成：{len(blocks)} 条 -> {OUTPUT_FILE}")
