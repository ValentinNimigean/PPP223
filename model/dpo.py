"""
Phase 3: DPO (Direct Preference Optimization) Template
Used to mathematically rank outputs to reduce hallucination.
Week 11 Deliverable.

WARNING: Requires 12GB+ NVIDIA VRAM. Do not execute on consumer ultra-books.
"""

import torch
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    from trl import DPOTrainer
    from datasets import load_dataset
except ImportError:
    print("Warning: Missing DPO dependencies. Make sure trl and datasets are installed.")

def run_dpo(model_id="Qwen/Qwen2.5-Coder-7B", dpo_data_path="preference_data.jsonl", output_dir="dpo_results"):
    print("Initializing DPO RLHF Pipeline...")
    
    # 1. Load Reference and Target Models
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # The reference model acts as the baseline probabilities to compare against
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    # 2. Setup LoRA (Apply RLHF efficiently)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    # 3. Load Preference Data
    # DPO requires 3 columns: "prompt", "chosen" (good code), "rejected" (hallucinations)
    print(f"Loading preference dataset from {dpo_data_path}...")
    # dataset = load_dataset("json", data_files=dpo_data_path)
    
    # 4. Configure DPO Trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        logging_steps=10,
        max_steps=200,
        remove_unused_columns=False,
        optim="paged_adamw_8bit"
    )

    # dpo_trainer = DPOTrainer(
    #     model,
    #     ref_model,
    #     args=training_args,
    #     beta=0.1, # KL penalty constraint (determines how far it can deviate from ref)
    #     train_dataset=dataset['train'],
    #     tokenizer=tokenizer,
    # )

    print("DPO Environment Ready. Ensure dataset contains 'prompt', 'chosen', and 'rejected' columns, then uncomment execution.")
    # dpo_trainer.train()

if __name__ == "__main__":
    print("DPO Module imported. Run run_dpo() when dataset is compiled.")
