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
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
except ImportError:
    print("Warning: Fine-tuning libraries not fully installed. Make sure to pip install unsloth trl datasets.")

def run_finetuning(model_id="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", dataset_path="my_curated_data.jsonl", output_dir="results"):
    print(f"Loading Base Model: {model_id} via Unsloth...")
    
    # Load model and tokenizer via Unsloth (this includes BitsAndBytes config internally)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Do NOT leave pad_token as <|endoftext|> for Qwen
    if tokenizer.pad_token == "<|endoftext|>" or tokenizer.pad_token is None:
        tokenizer.pad_token = "<|fim_pad|>"

    # Prepare for LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.0, # Unsloth optimizes for 0 dropout
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print(f"Trainable parameters set up for Unsloth.")

    print(f"Loading Dataset: {dataset_path}...")
    # Assume dataset has a conversational JSONL {"messages":[...]} format
    # e.g., dataset = load_dataset('json', data_files=dataset_path)

    cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True, 
        logging_steps=10, 
        save_strategy="epoch",
        max_length=2048,
        packing=False,
        assistant_only_loss=True,        # train on assistant turns only
        eos_token="<|im_end|>",          # required for Qwen
    )

    trainer = SFTTrainer(
        model=model, 
        args=cfg,
        # train_dataset=dataset['train'],
        processing_class=tokenizer,      # NOT tokenizer=
    )

    print("Starting QLoRA Training...")
    # trainer.train()
    
    # print("Saving adapter and merged GGUF...")
    # model.save_pretrained_merged(f"{output_dir}/merged", tokenizer, save_method="merged_16bit")
    # model.save_pretrained_gguf(f"{output_dir}/gguf", tokenizer, quantization_method="q4_k_m")

if __name__ == "__main__":
    print("Fine-tuning module is ready. Provide data and uncomment trainer.train() to begin.")
