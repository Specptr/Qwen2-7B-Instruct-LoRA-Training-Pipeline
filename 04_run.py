"""
实现交互式推理（对话生成）。支持 4bit 量化加载以降低显存占用，
并提供上下文截断与输出清理功能。

主要流程：
    1. 加载 tokenizer，并设置 pad_token
    2. 加载基础模型（支持 4bit 量化）
    3. 尝试加载 LoRA adapter（若存在）
    4. 构建 ChatML 格式的 prompt，自动截断超长上下文
    5. 使用 HuggingFace generate() 生成回复
    6. 清理生成结果（去除冗余语气词、重复符号）
    7. 提供交互式循环，支持用户输入与上下文记忆

输入：用户交互输入（命令行）
输出：模型生成的回复（实时打印到终端）

注意事项：
    - 若 LoRA adapter 未加载成功，将回退使用基础模型
    - 可通过修改参数（如 TEMPERATURE, TOP_P）调整生成风格
    - 输入 "exit"/"quit" 或 Ctrl+C 可退出交互

额外注释：
    - 我选用的格式为 模型输出“她：”
    - 会同时输出两行 其中第二行为自定义输出风格
    - 每次回复完 用户能选择是否加入history作为promt 选择加入则输入in
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
import re
import random

# 配置（按需修改）
BASE_MODEL = r"qwen2-7b-instruct"
ADAPTER_DIR = r"adapters/adapter"
USE_4BIT = True
MAX_CONTEXT_TOKENS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


GEN_MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9 # 越高越创造力 可能跑题
TOP_P = 0.9 # 越高越安全


print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


print("加载基础模型...")
model_kwargs = {"trust_remote_code": True}
if bnb_config is not None:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config,
        **model_kwargs
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        **model_kwargs
    )

# 尝试加载 LoRA adapter
use_adapter = False
if os.path.isdir(ADAPTER_DIR):
    try:
        print("尝试加载本地 LoRA adapter...")
        model = PeftModel.from_pretrained(model, ADAPTER_DIR, device_map="auto", local_files_only=True)
        use_adapter = True
        print("LoRA adapter 已加载:", ADAPTER_DIR)
    except Exception as e:
        print("加载 LoRA adapter 失败，降级使用基础模型。错误：", e)
else:
    print("请注意！请注意！你现在用的是默认模型！！！")

model.eval()

# 构建 prompt
def build_prompt_with_truncation(chat_history, tokenizer, max_tokens=MAX_CONTEXT_TOKENS):
    """
    将 chat_history 转为 ChatML prompt，
    并确保总 token 数 <= max_tokens by truncating older messages.
    """
    if not chat_history:
        return ""

    # 初始完整 prompt
    prompt = tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
    toks = tokenizer(prompt, return_tensors="pt", truncation=False)
    if toks["input_ids"].size(1) <= max_tokens:
        return prompt

    # 如果超长，丢掉最早的消息直到符合长度
    for start_idx in range(len(chat_history)):
        truncated = chat_history[start_idx:]
        prompt = tokenizer.apply_chat_template(truncated, tokenize=False, add_generation_prompt=True)
        toks = tokenizer(prompt, return_tensors="pt", truncation=False)
        if toks["input_ids"].size(1) <= max_tokens:
            return prompt

    # 最后硬截断
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    return tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)


# -----------------------
# 交互循环
# -----------------------
print("\n======")
print("waken up\n")
chat_history = []

# 自定义输出风格
def clean_response(text: str) -> str:
    # 删除“……”两字符省略号
    text = re.sub(r"……+", "…", text)
    # 删除连续3个及以上的句点
    text = re.sub(r"\.{3,}", " ", text)
    # 删除单独的语气词（避免误删其他词的一部分）
    text = re.sub(r"\b(嗯+|啊+|哦+|呀+|吧+|呢+)\b", " ", text)
    return text.strip()


while True:
    try:
        user_text = input("你：").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n\nOver")
        break
    if user_text.lower() in ("exit", "quit"):
        print("Over")
        break

    # append user
    chat_history.append({"role": "user", "content": user_text})

    # 构建 prompt
    prompt = build_prompt_with_truncation(chat_history, tokenizer, max_tokens=MAX_CONTEXT_TOKENS)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.2,       # 惩罚重复
            no_repeat_ngram_size=3,       # 禁止重复3-gram
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    gen_tokens = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    clean_resp = clean_response(response)
    if response:
        print("她：", response)
        print("她：", clean_resp)

        user_input = input("in? ").strip().lower()
        if user_input == 'in':
            chat_history.append({"role": "assistant", "content": clean_resp})
            print("read in")
        else:
            print("skip")
        print('\n')
    else:
        print("她：(未生成内容)")

print("她：bye")
