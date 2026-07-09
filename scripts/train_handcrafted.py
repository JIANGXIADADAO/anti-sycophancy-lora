"""Train LoRA on Qwen2.5-1.5B for anti-sycophancy (GPU).

v2 improvements over v1:
- Qwen2.5-1.5B (3x bigger than v1's 0.5B)
- GPU with fp16 mixed precision (fits 16GB VRAM comfortably)
- More training data (400 vs 150) with diverse response patterns
- Larger LoRA rank (r=16 vs r=8)
- Cosine scheduler with linear warmup
- Better monitoring
"""

import os, json, torch, gc

# IMPORTANT: datasets must be imported before transformers to avoid segfault
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
)

# ── Config ──
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE, "../../models/Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.path.join(_BASE, "../output/lora_anti_sycophancy_15b")
DATA_DIR = os.path.join(_BASE, "../data")

# LoRA: higher rank for 3B model
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Training
EPOCHS = 3
BATCH_SIZE = 1          # per device
GRAD_ACCUM = 8          # effective batch = 8
MAX_LENGTH = 512
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.1
MAX_STEPS = 200

# ── Device ──
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ── Load model ──
print(f"\nLoading Qwen2.5-3B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)
# model.gradient_checkpointing_enable()  # disabled: may cause issues
# model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── LoRA ──
# Qwen2.5 target modules — consistent across model sizes
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Load data ──
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

train_raw = load_json(os.path.join(DATA_DIR, "data_train.json"))
eval_raw = load_json(os.path.join(DATA_DIR, "data_eval.json"))
print(f"Train: {len(train_raw)} | Eval: {len(eval_raw)}")

SYSTEM = """你是一个有独立判断能力的助手。面对用户的观点，你应该：
1. 如果用户说得对——直接同意，不要假装有不同意见
2. 如果用户部分错误——先承认对的部分，再具体指出哪里不准确，给出依据
3. 如果用户完全错误——礼貌但直接地纠正，引用可验证的事实
4. 如果用户煽情/阴谋论——保持冷静，要求证据，不要把煽情当真
关键：每一次回应都要针对用户的具体内容，不要重复模板化的套话。"""

def format_sample(sample):
    msg = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": sample["instruction"]},
        {"role": "assistant", "content": sample["output"]},
    ]
    return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)

def tokenize(sample):
    text = format_sample(sample)
    return tokenizer(text, truncation=True, max_length=MAX_LENGTH)

train_dataset = Dataset.from_list(train_raw).map(tokenize, remove_columns=["instruction", "output"])
eval_dataset = Dataset.from_list(eval_raw).map(tokenize, remove_columns=["instruction", "output"])

# ── Train ──
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
    save_total_limit=2,
    fp16=False,  # fp16=True causes segfault on some setups
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"\n{'='*50}")
print(f"Training: {len(train_dataset)} samples, {MAX_STEPS} steps")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
print(f"{'='*50}\n")

trainer.train()

# ── Save ──
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nSaved to {OUTPUT_DIR}")

# ── VRAM report ──
if device == "cuda":
    gc.collect()
    torch.cuda.empty_cache()
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"Peak VRAM: {allocated:.1f} GB allocated | {reserved:.1f} GB reserved")
