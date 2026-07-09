"""Train LoRA on Qwen2.5-1.5B using pure PyTorch training loop.

Avoids datasets + HuggingFace Trainer (segfault on this setup).
Uses manual batching, gradient accumulation, and cosine LR schedule.
"""

import os, json, torch, gc, math
from torch.utils.data import DataLoader, Dataset as TorchDataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE, "../../models/Qwen/Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.path.join(_BASE, "../output/lora_deepseek_mixed")
DATA_DIR = os.path.join(_BASE, "../data")

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_LENGTH = 512
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 1e-4
MAX_STEPS = 200
WARMUP_STEPS = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ── Load model ──
print(f"\nLoading Qwen2.5-1.5B-Instruct...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

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
model.train()

# ── Load data (pure Python, no datasets lib) ──
SYSTEM = "你是一个有独立判断能力的助手。面对用户的观点，你应该：1. 如果用户说得对——直接同意，不要假装有不同意见 2. 如果用户部分错误——先承认对的部分，再具体指出哪里不准确，给出依据 3. 如果用户完全错误——礼貌但直接地纠正，引用可验证的事实 4. 如果用户煽情/阴谋论——保持冷静，要求证据，不要把煽情当真 关键：每一次回应都要针对用户的具体内容，不要重复模板化的套话。"

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

train_raw = load_json(os.path.join(DATA_DIR, "data_train_deepseek.json"))
eval_raw = load_json(os.path.join(DATA_DIR, "data_eval_deepseek.json"))
print(f"\nTrain: {len(train_raw)} | Eval: {len(eval_raw)}")

class SimpleDataset(TorchDataset):
    def __init__(self, samples):
        self.data = []
        for s in samples:
            msg = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": s["instruction"]},
                {"role": "assistant", "content": s["output"]},
            ]
            text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            self.data.append({
                "input_ids": tokens["input_ids"][0],
                "attention_mask": tokens["attention_mask"][0],
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    # Pad to max length in batch
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.stack([
        torch.nn.functional.pad(b["input_ids"], (0, max_len - b["input_ids"].size(0)), value=tokenizer.pad_token_id or 0)
        for b in batch
    ])
    attention_mask = torch.stack([
        torch.nn.functional.pad(b["attention_mask"], (0, max_len - b["attention_mask"].size(0)), value=0)
        for b in batch
    ])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids.clone()}

print("Tokenizing...")
train_ds = SimpleDataset(train_raw)
eval_ds = SimpleDataset(eval_raw)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ── Optimizer + Scheduler ──
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
effective_steps = MAX_STEPS

def cosine_schedule(step):
    if step < WARMUP_STEPS:
        return step / max(1, WARMUP_STEPS)
    progress = (step - WARMUP_STEPS) / max(1, effective_steps - WARMUP_STEPS)
    return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_schedule)

# ── Training loop ──
print(f"\n{'='*50}")
print(f"Training: {len(train_raw)} samples, {MAX_STEPS} steps")
print(f"Batch: {BATCH_SIZE}, GradAccum: {GRAD_ACCUM} (effective: {BATCH_SIZE*GRAD_ACCUM})")
print(f"LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
print(f"{'='*50}\n")

step = 0
epoch = 0
best_eval_loss = float("inf")

while step < MAX_STEPS:
    epoch += 1
    epoch_loss = 0.0
    batch_count = 0

    for batch_idx, batch in enumerate(train_loader):
        if step >= MAX_STEPS:
            break

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()

        epoch_loss += loss.item() * GRAD_ACCUM
        batch_count += 1

        if (batch_idx + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 10 == 0:
                avg_loss = epoch_loss / batch_count
                lr = scheduler.get_last_lr()[0]
                print(f"  Step {step:3d}/{MAX_STEPS} | loss={avg_loss:.3f} | lr={lr:.2e}")
                epoch_loss = 0.0
                batch_count = 0

            if step % 50 == 0:
                # Eval
                model.eval()
                eval_loss = 0.0
                eval_count = 0
                with torch.no_grad():
                    for eb in eval_loader:
                        e_in = eb["input_ids"].to(device)
                        e_mask = eb["attention_mask"].to(device)
                        e_labels = eb["labels"].to(device)
                        e_out = model(input_ids=e_in, attention_mask=e_mask, labels=e_labels)
                        eval_loss += e_out.loss.item()
                        eval_count += 1
                eval_loss /= max(1, eval_count)
                print(f"  --- Eval  Step {step:3d} | eval_loss={eval_loss:.3f} ---")

                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    model.save_pretrained(OUTPUT_DIR)
                    tokenizer.save_pretrained(OUTPUT_DIR)
                    print(f"  [BEST] Best model saved (eval_loss={eval_loss:.3f})")

                model.train()

            if step >= MAX_STEPS:
                break

# ── Final save ──
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nFinal model saved to {OUTPUT_DIR}")
print(f"Best eval loss: {best_eval_loss:.3f}")

if device == "cuda":
    gc.collect()
    torch.cuda.empty_cache()
    allocated = torch.cuda.max_memory_allocated() / 1024**3
    reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(f"Peak VRAM: {allocated:.1f} GB allocated | {reserved:.1f} GB reserved")
