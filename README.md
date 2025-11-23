# Qwen2-7B-Instruct-LoRA-Training-Pipeline
Developed a reproducible LoRA fine-tuning pipeline for Qwen2-7B-Instruct using HuggingFace Transformers, including data preprocessing, training scripts, and experimental analysis.
---

本项目展示在本地环境中：
下载 Qwen2-7B-Instruct 模型，
使用 LoRA（QLoRA）进行轻量微调，
保存并加载训练得到的 LoRA 适配器，
实现完全离线、本地 GPU 推理，
使用一块 12GB 显存的显卡完成训练。

---

## 项目概述
- **模型**：Qwen2-7B-Instruct  
- **方法**：LoRA  
- **硬件环境**：RTX 4080 Laptop GPU  
- **精度设置**：bf16 / fp16 混合精度  
- **主要功能**：
  - 原始语料转 JSONL 格式
  - 使用 HuggingFace tokenizer 进行分词
  - 自定义 Trainer，支持 LoRA 微调
  - 训练与评估脚本
  - 实验笔记与曲线分析
 
## 参数配置
注：适用于RTX 4080 Laptop测试使用
### Lora:
- r = 16
- lora_alpha = 32
- target_modules = ["q_proj", "v_proj"]
- lora_dropout = 0.05
- bias = "none"
- task_type = "CAUSAL_LM"
### 训练参数:
- per_device_train_batch_size = 1
- gradient_accumulation_steps = 4
- num_train_epochs = 3
- learning_rate = 2e-4
- logging_steps = 10
- fp16 = True
- save_strategy = "epoch"
- save_total_limit = 2
- report_to = "none"


