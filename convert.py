from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
lora_dir = "./finetuned_models/final_model"
output_dir = "./merged_model"
# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
# 合并LoRA权重
model = PeftModel.from_pretrained(model, lora_dir)
model = model.merge_and_unload() # 关键合并操作
# 保存完整模型
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
