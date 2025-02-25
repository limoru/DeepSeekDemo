#!/usr/bin/env python3
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import argparse
import torch
import os

def main(args):
    # 1. 初始化配置
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.fp16 else torch.float32
    )

    # 3. 加载数据集
    dataset = load_dataset('json', data_files=args.dataset_path)

    # 4. 数据预处理
    def tokenize_function(examples):
        # 添加EOS token确保生成完整性
        text = [p + "\n" + r + tokenizer.eos_token for p, r in zip(examples["prompt"], examples["response"])]
        
        tokenized = tokenizer(
            text, 
            truncation=True, 
            max_length=args.max_length,
            padding="max_length"
        )
        # 显式设置labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # 5. 配置LoRA
    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=args.lora_dropout,
            task_type="CAUSAL_LM"
         )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 6. 训练参数配置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=args.fp16,
        logging_steps=10,
        save_steps=500,
        report_to="none"
    )

    # 7. 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )
    trainer.train()

    # 8. 保存模型
    final_model_dir = os.path.join(args.output_dir, "final_model")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"模型已保存至：{final_model_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 必需参数
    parser.add_argument("--dataset_path", type=str, default="data/my_dataset.json")
    parser.add_argument("--output_dir", type=str, default="finetuned_models")

    # 模型参数
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")

    # LoRA参数
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()
    main(args)
