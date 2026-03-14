#!/usr/bin/env python3
# train.py — runs INSIDE the SageMaker training job on ml.g4dn.xlarge
# DO NOT run this locally. It is uploaded to S3 and executed by SageMaker.

import os
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

logger.info("=== Vanimitra Training Job Starting ===")
logger.info(f"Python: {sys.version}")

# ── Install dependencies ──────────────────────────────────────────────────────
os.system("pip install -q transformers==4.40.0 peft==0.10.0 trl==0.8.6 datasets==2.19.0 accelerate==0.30.0 bitsandbytes==0.43.1")

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "/opt/ml/model"
TRAIN_DATA_PATH = "/opt/ml/input/data/train/vanimitra_train.jsonl"

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    logger.info("HF token found, logging in...")
    os.system(f"huggingface-cli login --token {HF_TOKEN}")
else:
    logger.warning("No HF_TOKEN env var found — model download may fail if gated")

# ── Load dataset ──────────────────────────────────────────────────────────────
logger.info(f"Loading dataset from {TRAIN_DATA_PATH}")
records = []
with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

logger.info(f"Loaded {len(records)} training examples")
dataset = Dataset.from_list(records)

# ── Load tokenizer ────────────────────────────────────────────────────────────
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN if HF_TOKEN else None,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── Load model ────────────────────────────────────────────────────────────────
logger.info("Loading base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN if HF_TOKEN else None,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

logger.info("Model loaded successfully")

# ── LoRA config ───────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Format function ───────────────────────────────────────────────────────────
def format_chat(example):
    """Apply Qwen chat template to messages list."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

logger.info("Formatting dataset with chat template...")
dataset = dataset.map(format_chat, remove_columns=["messages"])
logger.info(f"Sample formatted example:\n{dataset[0]['text'][:300]}")

# ── Training arguments ────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
    dataloader_num_workers=0,
    remove_unused_columns=False,
    optim="paged_adamw_8bit",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=512,
    tokenizer=tokenizer,
    packing=False,
)

logger.info("=== Starting Training ===")
trainer.train()
logger.info("=== Training Complete ===")

# ── Save LoRA adapters ────────────────────────────────────────────────────────
logger.info(f"Saving LoRA adapters to {OUTPUT_DIR}")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save base model name for merge step
with open(os.path.join(OUTPUT_DIR, "base_model_name.txt"), "w") as f:
    f.write(MODEL_ID)

logger.info("=== Job complete. Adapters saved. ===")
