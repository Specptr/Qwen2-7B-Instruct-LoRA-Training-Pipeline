# 03_train_lora.py
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_PATH   = r"qwen2-7b-instruct"
DATASET_PATH = r"data/tokenized"
OUTPUT_PATH  = r"adapters/qwen2-lora"

# --------- 4bit ---------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)

# --------- LoRA ---------
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        # attention
        "q_proj", "k_proj", "v_proj", "o_proj",
        # mlp（Qwen2 常见命名）
        "gate_proj", "up_proj", "down_proj",
    ]
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------- dataset ---------
ds = load_from_disk(DATASET_PATH)

# --------- collator: 保留 labels ---------
def collate_fn(batch):
    # batch: list of dicts with input_ids/attention_mask/labels
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq, pad_val):
        return seq + [pad_val] * (max_len - len(seq))

    input_ids = [pad(x["input_ids"], tokenizer.pad_token_id) for x in batch]
    attention_mask = [pad(x["attention_mask"], 0) for x in batch]
    labels = [pad(x["labels"], -100) for x in batch]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

# --------- training args ---------
args = TrainingArguments(
    output_dir=OUTPUT_PATH,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=1e-4,
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="none",
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds,
    data_collator=collate_fn,
)

trainer.train()

os.makedirs(OUTPUT_PATH, exist_ok=True)
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)  # 可选：方便复现
print(f"完成：adapter -> {OUTPUT_PATH}")
