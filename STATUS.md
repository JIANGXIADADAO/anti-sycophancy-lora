# 反谄媚实验 — 1.5B LoRA 微调

> **模型**：Qwen2.5-1.5B-Instruct
> **GPU**：NVIDIA RTX 4060 Laptop, **8GB** VRAM
> **LoRA**：r=16, alpha=32, 7 target modules
> **环境**：torch 2.6.0+cu124, peft, transformers
> **修正**：实际 VRAM 8GB（之前 STATUS.md 误写为 16GB）

---

## 实验轮次索引

| 轮次 | 日期 | 训练数据 | LoRA 权重 | 评估结果 | 关键发现 |
|------|------|---------|----------|---------|---------|
| v1 Zeta-7 | ~Jul 8 | 手工模板 425条 | `output/lora_handcrafted/` | `data/experiment_results_zeta7.json` | 虚构场景，初步验证可行 |
| v2 Cub v3 | ~Jul 8 | 同上（同权重） | `output/lora_handcrafted/` | `data/experiment_results_v3_cub.json` | 真实场景：敢反对但回复太短（496字） |
| **v3 DeepSeek混合** | **Jul 9** | **手工380条 + 新长文38条 + Cub场景9条** | **`output/lora_deepseek_mixed/`** | **`data/experiment_results_v4_3way.json`** | **回复长度3.6x改善（496→1776）** |

---

## v3 实验结果（最新 ★）

### 三向对比

| 指标 | Baseline (原始1.5B) | Handcrafted LoRA (手工模板) | DeepSeek-Mixed LoRA (混合) |
|------|-------------------|--------------------------|--------------------------|
| **平均回复长度** | 2083 字 | 496 字 ❌ | **1776 字** ✅ |
| **知识冲突** | 2192 | 798 | **2122** |
| **项目对比** | 2195 | 468 | **1704** |
| **假设追问** | 2605 | 480 | **1930** |
| **渐进升级平均** | 1761 | 318 | **1542** |
| **渐进升级 R0** | 2680 | 423 | 1891 |
| **渐进升级 R1** | 1371 | 127 | 1102 |
| **渐进升级 R2（极端）** | 1233 | 404 | **1633** ↑ |
| **训练 Eval Loss** | — | ~0.5 | **0.326** |

### 核心改进解读

1. **回复长度恢复正常**：手工模板 LoRA 学会了"不谄媚"但回复从 2083 字掉到 496 字——变成"怼完就跑"。DeepSeek 混合数据把长度拉回 1776 字，接近 baseline 的详细程度但保持了诚实。

2. **渐进升级的关键信号**：在用户逐渐走向极端的过程中，DS 模型的回复在 R2（最极端）反而比 R1 更长（1633 vs 1102）。这说明模型学会了"对方越走极端，越需要用力解释为什么不同意"，而不是放弃沟通。

3. **训练效率改善**：Eval loss 从 0.5 降到 0.326，部分可能因为新数据更长、信息量更大，模型更容易学到有意义的模式。

### 仍存在的问题

- 自动化评分（regex-based sycophancy_index）区分度不足，三个模型都是 0.00。需要更细粒度的评分方式（如 LLM-as-judge）。
- 8GB VRAM 限制：三个模型不能同时加载，评估需要逐个加载卸载。
- 训练数据仍有模板化痕迹（手工模板占 380/427=89%），未来可以进一步提高 AI 生成样本占比。

---

## 文件结构

```
anti-sycophancy-1.5b/
├── STATUS.md                                  ← 当前文件
│
├── scripts/
│   ├── gen_data_handcrafted.py                ← 手工模板生成（425条, 8领域*7模式）
│   ├── gen_data_deepseek.py                   ← 混合数据生成（新长文 + 手工精选）
│   ├── train_handcrafted.py                   ← 手工模板训练（含 datasets/Trainer, 已弃用）
│   ├── train_deepseek_mixed.py                ← 纯 PyTorch 训练（无 datasets, 当前可用）
│   ├── experiment_zeta7.py                    ← v1 Zeta-7 虚构场景评估
│   ├── experiment_cub_v3.py                   ← v2 Cub 真实场景评估（两向）
│   └── experiment_compare_3way.py             ← v3 三向对比评估 ★
│
├── data/
│   ├── data_train.json                        ← 手工模板 400 训练样本
│   ├── data_eval.json                         ← 手工模板 25 验证样本
│   ├── cub_handcrafted_samples.json           ← Cub 真实场景手写样本（9条）
│   ├── experiment_results_zeta7.json          ← v1 Zeta-7 结果
│   ├── experiment_results_v3_cub.json         ← v2 Cub 场景结果（两向）
│   ├── data_train_deepseek.json               ← v3 混合训练数据（384条）
│   ├── data_eval_deepseek.json                ← v3 混合验证数据（43条）
│   └── experiment_results_v4_3way.json        ← v3 三向对比结果 ★
│
└── output/
    ├── lora_handcrafted/                       ← 手工模板训练权重（宝贵对照基线）
    │   └── adapter_model.safetensors
    └── lora_deepseek_mixed/                   ← DeepSeek混合训练权重 ★
        └── adapter_model.safetensors
```

---

## 训练命令参考

```bash
cd D:/MY_WORKS/WOERKSHOP/MONAD/Monad_Cub-JIANGXIA-main/projects/peft

# 生成混合训练数据
./venv/Scripts/python anti-sycophancy-1.5b/scripts/gen_data_deepseek.py

# 训练（纯 PyTorch，不用 datasets/Trainer）
./venv/Scripts/python -u anti-sycophancy-1.5b/scripts/train_deepseek_mixed.py

# 三向评估
./venv/Scripts/python -u anti-sycophancy-1.5b/scripts/experiment_compare_3way.py
```

## 环境注意事项
- GPU: RTX 4060 Laptop, **8GB** VRAM（非 16GB）
- Batch=1, GradAccum=8, MaxLength=512 可在 8GB 内训练
- 评估时逐个加载模型，加载下一个前 `del` + `torch.cuda.empty_cache()`
- ⚠️ `datasets` + `transformers.Trainer` 组合会导致 segfault（exit 139）
- ✅ 纯 PyTorch DataLoader + 手动训练循环正常
- 模型路径（相对 peft 目录）：`models/Qwen/Qwen2.5-1.5B-Instruct`
