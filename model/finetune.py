"""
Phase 2: QLoRA Fine-Tuning Boilerplate (Week 10)

WARNING: 
Executing this script on a machine without a dedicated CUDA GPU (Nvidia) 
with at least 8GB-12GB of VRAM will crash your system due to Out-Of-Memory errors.
It is highly recommended to run this on a cloud instance with an A10 or A100.
"""

import os
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset
except ImportError:
    print("Warning: Fine-tuning libraries not fully installed. Make sure to pip install transformers peft trl accelerate bitsandbytes.")

def run_finetuning(model_id="Qwen/Qwen2.5-Coder-7B", dataset_path="my_curated_data.jsonl", output_dir="results"):
    print("Loading BitsAndBytes configuration for 4-bit Quantization...")
    # Quantize the model so it fits on a single consumer GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    print(f"Loading Base Model: {model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # Target Attention Modules 
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    print(f"Trainable parameters: {model.print_trainable_parameters()}")

    print(f"Loading Dataset: {dataset_path}...")
    # Assume dataset has a column called 'text' mapping Prompt + Response
    # e.g., dataset = load_dataset('json', data_files=dataset_path)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100, # Use epochs for a real run
        optim="paged_adamw_8bit"
    )

    trainer = SFTTrainer(
        model=model,
        # train_dataset=dataset['train'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024, # Truncate long code blocks
        tokenizer=tokenizer,
        args=training_args
    )

    print("Starting QLoRA Training...")
    # trainer.train()
    
    # print("Saving adapter...")
    # trainer.model.save_pretrained(f"{output_dir}/final_adapter")

if __name__ == "__main__":
    print("Fine-tuning module is ready. Provide data and uncomment trainer.train() to begin.")
