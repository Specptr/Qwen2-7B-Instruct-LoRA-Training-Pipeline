# run_infer.py
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -----------------------
# 配置（按需修改）
# -----------------------
BASE_MODEL = r"qwen2-7b-instruct"
ADAPTER_DIR = r"adapters/qwen2-lora"
USE_4BIT = True
MAX_CONTEXT_TOKENS = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# generation params (default)
GEN_MAX_NEW_TOKENS = 256
TEMPERATURE = 0.9
TOP_P = 0.9

# identity stabilization (only when identity intent detected)
ID_TEMPERATURE = 0.2
ID_TOP_P = 0.9
ID_MAX_NEW_TOKENS = 64

# -----------------------
# System message（入口级锚点）
# -----------------------
SYSTEM_MSG = {
    "role": "system",
    "content": (
        "约定：你的名字是xx。\n"
        "当用户询问你的身份/名字/你是谁/自我介绍/怎么称呼你时，只回答：我是xx。\n"
        "除非被询问，否则不要主动提及名字或身份。\n"
        "整体语气：..."
    )
}

# -----------------------
# 加载 tokenizer
# -----------------------
print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -----------------------
# BitsAndBytes 配置
# -----------------------
bnb_config = None
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

# -----------------------
# 加载基础模型
# -----------------------
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

# -----------------------
# 尝试加载 LoRA adapter
# -----------------------
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
    print("请注意！你现在用的是默认模型（未找到 adapter 目录）。")

model.eval()

# -----------------------
# 工具：身份意图检测
# -----------------------
_ID_PAT = re.compile(
    r"(你是谁|你叫(什么|啥)|你的名字|怎么称呼你|自我介绍|介绍一下你|你是什么|who\s*are\s*you|what('?s| is)\s*your\s*name)",
    re.I
)

def is_identity_query(text: str) -> bool:
    return _ID_PAT.search(text.strip()) is not None

# -----------------------
# 清洗
# -----------------------
def clean_response(text: str) -> str:
    text = re.sub(r"……+", "…", text)
    text = re.sub(r"\.{3,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# -----------------------
# 构建 prompt（永远保留 system，并在截断时保护它）
# -----------------------
def build_prompt_with_truncation(body_history, tokenizer, max_tokens=MAX_CONTEXT_TOKENS):
    """
    body_history: 不包含 system 的 user/assistant 列表
    返回：拼好且不超长的 prompt（system 永远在最前）
    """
    def render(msgs):
        p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        t = tokenizer(p, return_tensors="pt", truncation=False)
        return p, t["input_ids"].size(1)

    msgs_full = [SYSTEM_MSG] + body_history
    prompt, n = render(msgs_full)
    if n <= max_tokens:
        return prompt

    # 截断：只截 body_history，system 永不丢
    for start_idx in range(len(body_history)):
        truncated = body_history[start_idx:]
        prompt, n = render([SYSTEM_MSG] + truncated)
        if n <= max_tokens:
            return prompt

    # 兜底硬截断（仍然包含 system）
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens)
    return tokenizer.decode(enc["input_ids"][0], skip_special_tokens=False)

# -----------------------
# 交互循环
# -----------------------
print("\n======")
print("waken up\n")
history = []  # 仅存 user/assistant，不存 system

HELP = (
    "命令：\n"
    "  /reset  清空历史\n"
    "  /undo   撤销上一条 user（以及可能的 assistant）\n"
    "  /show   显示当前历史条数\n"
    "写回：\n"
    "  in      写回原始回复\n"
    "  in_c    写回清洗后的回复\n"
    "  skip    不写回（默认）\n"
)

while True:
    try:
        user_text = input("你：").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\nOver")
        break

    if not user_text:
        continue

    # 快捷命令
    if user_text.lower() in ("exit", "quit"):
        print("Over")
        break
    if user_text.lower() == "/help":
        print(HELP)
        continue
    if user_text.lower() == "/reset":
        history.clear()
        print("（已清空历史）\n")
        continue
    if user_text.lower() == "/show":
        print(f"（当前历史条数：{len(history)}）\n")
        continue
    if user_text.lower() == "/undo":
        if history:
            popped = history.pop()
            # 如果最后一条是 assistant，继续弹出对应 user（尽量对齐）
            if popped.get("role") == "assistant" and history and history[-1].get("role") == "user":
                history.pop()
            print("（已撤销上一轮）\n")
        else:
            print("（历史为空）\n")
        continue

    # append user
    history.append({"role": "user", "content": user_text})

    # prompt
    prompt = build_prompt_with_truncation(history, tokenizer, max_tokens=MAX_CONTEXT_TOKENS)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # generation params (conditional)
    if is_identity_query(user_text):
        temperature = ID_TEMPERATURE
        top_p = ID_TOP_P
        max_new = ID_MAX_NEW_TOKENS
    else:
        temperature = TEMPERATURE
        top_p = TOP_P
        max_new = GEN_MAX_NEW_TOKENS

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    if not response:
        print("：(未生成内容)\n")
        # 如果没生成，撤销刚才那条 user，避免污染历史
        if history and history[-1]["role"] == "user":
            history.pop()
        continue

    print("：", response)

    # 你决定是否写回
    ctrl = input("\n ").strip().lower()
    if ctrl == "in":
        history.append({"role": "assistant", "content": response})
        print("read in\n")
    elif ctrl == "in_c":
        history.append({"role": "assistant", "content": clean_response(response)})
        print("read in (clean)\n")
    else:
        # 默认 skip：撤销最后一条 user，保证“采样不写回=不影响下一轮”
        # 这与你原意一致：只用来挑选满意输出
        if history and history[-1]["role"] == "user":
            history.pop()
        print("skip\n")
