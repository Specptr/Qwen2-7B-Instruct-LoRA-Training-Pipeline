"""
功能说明：
    本脚本用于在 Qwen2-7B-Instruct 模型上进行 LoRA 微调训练。
    采用 HuggingFace Transformers 与 PEFT 库，结合 4bit 量化加载，

主要流程：
    1. 自定义 Trainer，支持 labels 并计算因果语言模型的交叉熵损失
    2. 加载预训练模型与 Tokenizer，配置 4bit 量化
    3. 准备 LoRA 配置（r=16, alpha=32, dropout=0.05，作用于 q_proj/v_proj）
    4. 加载已分词的数据集（tokenized dataset）
    5. 设置 DataCollator，实现动态 padding
    6. 配置训练参数
    7. 启动训练并保存 LoRA 权重

输入已分词的数据集目录
输出LoRA 微调后的权重

可根据硬件条件调整 batch size 与梯度累积步数
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
import torch
import torch.nn.functional as F

MODEL_PATH = "qwen2-7b-instruct"
DATASET_PATH = "data/tokenized"
OUTPUT_PATH = "adapters/adapter"



class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs["input_ids"]
        outputs = model(**inputs)
        logits = outputs.get("logits")
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        return (loss, outputs) if return_outputs else loss


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)


model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)


tokenized_dataset = load_from_disk(DATASET_PATH)



data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


training_args = TrainingArguments(
    per_device_train_batch_size=1,          
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    fp16=True,
    output_dir=OUTPUT_PATH,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    save_on_each_node=True
)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()


model.save_pretrained(training_args.output_dir)
print(f"LoRA 权重已保存到 {training_args.output_dir}")
