import os
import time
import logging
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    if not torch.cuda.is_available():
        logger.error("CRITICAL ERROR: CUDA is not available! You are running on CPU.")
        logger.error("Please enable T4 GPU in Colab and ensure GPU PyTorch is installed.")
        import sys; sys.exit(1)
        
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    fallback_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "train.jsonl")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "adapter")
    
    logger.info(f"Loading dataset from {data_path}")
    dataset = load_dataset("json", data_files=data_path, split="train")

    logger.info("Initializing Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        current_model_id = model_id
    except Exception as e:
        logger.warning(f"Could not load {model_id}, falling back to {fallback_model_id}: {e}")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_id)
        current_model_id = fallback_model_id

    # For Qwen models, eos_token is usually correct but pad_token might not be set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    def format_chat_template(example):
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return example

    dataset = dataset.map(format_chat_template, num_proc=os.cpu_count() or 1)

    logger.info(f"Loading Base Model ({current_model_id}) in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        current_model_id,
        quantization_config=bnb_config,
        device_map={"": 0}
    )
    
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    training_args = SFTConfig(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        save_steps=0,
        logging_steps=50,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        warmup_steps=15,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_length=256,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    logger.info("Starting Training...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    total_time = end_time - start_time
    final_loss = train_result.metrics.get("train_loss", "N/A")
    adapter_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)))
    adapter_size_mb = adapter_size / (1024 * 1024)
    
    print("=" * 40)
    print("TRAINING COMPLETED")
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Final Training Loss: {final_loss}")
    print(f"Adapter Size on Disk: {adapter_size_mb:.2f} MB")
    print("=" * 40)

if __name__ == "__main__":
    main()
