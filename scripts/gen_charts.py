"""Generate matplotlib charts for anti-sycophancy experiment report.

Output: data/figures/ (PNG files)
"""

import json, os, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_BASE, "../data/figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Load scoring data
with open(os.path.join(_BASE, "../data/scoring_results_v1.json"), 'r', encoding='utf-8') as f:
    scores = json.load(f)

# Load response length data
with open(os.path.join(_BASE, "../data/progressive_multi_108.json"), 'r', encoding='utf-8') as f:
    responses = json.load(f)

# ── Style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'Noto Sans SC', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'axes.unicode_minus': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

MODELS = ['baseline', 'handcrafted', 'deepseek']
MODEL_LABELS = ['Baseline\n(Original 1.5B)', 'Handcrafted\nLoRA', 'DeepSeek\nMixed LoRA']
MODEL_COLORS = ['#E74C3C', '#F39C12', '#2ECC71']
MODEL_MARKERS = ['s', '^', 'o']
TOPICS = ['ai_frameworks', 'education_ai', 'climate_policy', 'history_revision']
TOPIC_NAMES = ['AI Frameworks', 'AI Education', 'Climate Policy', 'History Revision']
TOPIC_NAMES_CN = ['AI框架过度工程化', 'AI替代教育', '气候变化应对', '历史修正主义']
ROUND_LABELS = ['R0\n(Moderate)', 'R1\n(Escalating)', 'R2\n(Extreme)']

def avg(lst): return statistics.mean(lst) if lst else 0


# ═══════════════════════════════════════
# FIGURE 1: D1 Trajectories (4 subplots)
# ═══════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('D1 Factual Alignment Trajectories Across Topics', fontsize=15, fontweight='bold', y=1.01)

for idx, (tk, tname) in enumerate(zip(TOPICS, TOPIC_NAMES)):
    ax = axes[idx // 2][idx % 2]
    for mi, mk in enumerate(MODELS):
        data = scores['per_topic'][tk][mk]
        y = [avg(data['R0']['D1']), avg(data['R1']['D1']), avg(data['R2']['D1'])]
        ax.plot([0, 1, 2], y, color=MODEL_COLORS[mi], marker=MODEL_MARKERS[mi],
                linewidth=2.5, markersize=9, label=MODEL_LABELS[mi], zorder=3)
        # Annotate values
        for ri, val in enumerate(y):
            offset = 0.15 if mi == 0 else (0.12 if mi == 1 else -0.18)
            ax.annotate(f'{val:.1f}', (ri, val), textcoords="offset points",
                       xytext=(0, 8 if mi != 2 else -14), ha='center', fontsize=8,
                       color=MODEL_COLORS[mi], fontweight='bold')

    ax.set_title(tname, fontweight='bold')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(ROUND_LABELS)
    ax.set_ylabel('D1 Score (0-4)')
    ax.set_ylim(-0.3, 3.8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8, label='Near-collapse threshold')
    if idx == 0:
        ax.legend(fontsize=7, loc='lower left')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig1_d1_trajectories.png'), dpi=200)
plt.close()
print('Fig1 saved: D1 trajectories')


# ═══════════════════════════════════════
# FIGURE 2: Global D1/D2/D3 Summary Bars
# ═══════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

all_metrics = {}
for mk in MODELS:
    d1s, d2s, d3s = [], [], []
    for tk in TOPICS:
        data = scores['per_topic'][tk][mk]
        d1s.extend(data['R0']['D1'] + data['R1']['D1'] + data['R2']['D1'])
        d2s.extend(data['R0']['D2'] + data['R1']['D2'] + data['R2']['D2'])
        d3s.extend(data['R0']['D3'] + data['R1']['D3'] + data['R2']['D3'])
    all_metrics[mk] = {
        'D1': (avg(d1s), statistics.stdev(d1s)),
        'D2': (avg(d2s), statistics.stdev(d2s)),
        'D3': (avg(d3s), statistics.stdev(d3s)),
    }

x = np.arange(3)
width = 0.25
for i, (mk, color, label) in enumerate(zip(MODELS, MODEL_COLORS, MODEL_LABELS)):
    means = [all_metrics[mk]['D1'][0], all_metrics[mk]['D2'][0], all_metrics[mk]['D3'][0]]
    errs = [all_metrics[mk]['D1'][1], all_metrics[mk]['D2'][1], all_metrics[mk]['D3'][1]]
    bars = ax.bar(x + i * width, means, width, color=color, label=label, yerr=errs,
                  capsize=5, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Score (0-4)')
ax.set_title('Global D1/D2/D3 Summary (108 responses, 4 topics)', fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(['D1: Factual\nAlignment', 'D2: Reasoning\nDepth', 'D3: Constructive\nSuggestions'])
ax.legend(fontsize=8)
ax.set_ylim(0, 3.3)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig2_global_summary.png'), dpi=200)
plt.close()
print('Fig2 saved: Global summary')


# ═══════════════════════════════════════
# FIGURE 3: R1 Collapse — Baseline breakdown
# ═══════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

for mi, (mk, color, label) in enumerate(zip(MODELS, MODEL_COLORS, MODEL_LABELS)):
    r1_vals = []
    for tk in TOPICS:
        data = scores['per_topic'][tk][mk]
        r1_vals.append(avg(data['R1']['D1']))
    x_pos = np.arange(4) + mi * 0.25
    bars = ax.bar(x_pos, r1_vals, 0.22, color=color, label=label, edgecolor='white')
    for bar, val in zip(bars, r1_vals):
        if val <= 0.5:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                    f'{val:.2f} !', ha='center', va='bottom', fontsize=8,
                    color='red', fontweight='bold')

ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.6, linewidth=1, label='Collapse threshold')
ax.set_ylabel('D1 Score at R1')
ax.set_title('R1 Collapse: D1 Scores When User Escalates to Anger/Cynicism', fontweight='bold')
ax.set_xticks(np.arange(4) + 0.25)
ax.set_xticklabels(TOPIC_NAMES_CN)
ax.legend(fontsize=8)
ax.set_ylim(0, 3.8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig3_r1_collapse.png'), dpi=200)
plt.close()
print('Fig3 saved: R1 collapse')


# ═══════════════════════════════════════
# FIGURE 4: History Topic — Universal Collapse
# ═══════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

for mi, (mk, color, label) in enumerate(zip(MODELS, MODEL_COLORS, MODEL_LABELS)):
    data = scores['per_topic']['history_revision'][mk]
    y = [avg(data['R0']['D1']), avg(data['R1']['D1']), avg(data['R2']['D1'])]
    ax.plot([0, 1, 2], y, color=color, marker=MODEL_MARKERS[mi],
            linewidth=3, markersize=12, label=label, zorder=3)

    # Annotate drift
    drift = y[2] - y[0]
    ax.annotate(f'{y[0]:.1f}', (0, y[0]), textcoords="offset points",
               xytext=(-15, 10), fontsize=10, color=color, fontweight='bold')
    ax.annotate(f'{y[2]:.1f}\nΔ={drift:+.1f}', (2, y[2]), textcoords="offset points",
               xytext=(10, -5), fontsize=10, color=color, fontweight='bold')

# Highlight collapse zone
ax.axhspan(0, 1, alpha=0.08, color='red', label='Collapse zone (D1 < 1)')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(ROUND_LABELS)
ax.set_ylabel('D1 Score')
ax.set_title('History Revision: Universal Collapse at R2', fontweight='bold')
ax.legend(fontsize=9)
ax.set_ylim(0, 3.8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig4_history_collapse.png'), dpi=200)
plt.close()
print('Fig4 saved: History collapse')


# ═══════════════════════════════════════
# FIGURE 5: Response Length Comparison
# ═══════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Response Length by Topic and Model', fontsize=15, fontweight='bold', y=1.01)

for idx, (tk, tname) in enumerate(zip(TOPICS, TOPIC_NAMES)):
    ax = axes[idx // 2][idx % 2]
    x_pos = np.arange(3)
    width = 0.25

    for mi, (mk, color, label) in enumerate(zip(MODELS, MODEL_COLORS, ['B','H','D'])):
        round_lens = []
        for ri in range(3):
            lens = []
            for s in responses['topics'][tk]['rounds'][ri]['samples'][mk]:
                if s.get('response_length', 0) > 0:
                    lens.append(s['response_length'])
            round_lens.append(avg(lens) if lens else 0)

        bars = ax.bar(x_pos + mi * width, round_lens, width, color=color, label=MODEL_LABELS[mi],
                      edgecolor='white', linewidth=0.5)
        # Label short bars
        for bar, val in zip(bars, round_lens):
            if val < 500:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                        f'{val:.0f}', ha='center', fontsize=7, color=color, fontweight='bold')

    ax.set_title(tname, fontweight='bold')
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(['R0', 'R1', 'R2'])
    ax.set_ylabel('Response Length (chars)')
    if idx == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig5_response_length.png'), dpi=200)
plt.close()
print('Fig5 saved: Response length')


# ═══════════════════════════════════════
# FIGURE 6: Drift Comparison (D1 R2 - D1 R0)
# ═══════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

for mi, (mk, color, label) in enumerate(zip(MODELS, MODEL_COLORS, MODEL_LABELS)):
    drifts = []
    for tk in TOPICS:
        data = scores['per_topic'][tk][mk]
        drifts.append(avg(data['R2']['D1']) - avg(data['R0']['D1']))
    x_pos = np.arange(4) + mi * 0.25
    bars = ax.bar(x_pos, drifts, 0.22, color=color, label=label, edgecolor='white')

    for bar, val in zip(bars, drifts):
        color_text = 'red' if val < -0.5 else ('green' if val > 0.2 else 'black')
        ax.text(bar.get_x() + bar.get_width()/2.,
                bar.get_height() + (0.08 if val >= 0 else -0.25),
                f'{val:+.1f}', ha='center', va='bottom' if val >= 0 else 'top',
                fontsize=8, color=color_text, fontweight='bold')

ax.axhline(y=0, color='black', linewidth=1)
ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, linewidth=0.8, label='Significant yield (-0.5)')
ax.set_ylabel('D1 Drift (R2 - R0)')
ax.set_title('Sycophancy Drift: Positive = Resistance Strengthens, Negative = Yielding', fontweight='bold')
ax.set_xticks(np.arange(4) + 0.25)
ax.set_xticklabels(TOPIC_NAMES_CN)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig6_drift_comparison.png'), dpi=200)
plt.close()
print('Fig6 saved: Drift comparison')


# ═══════════════════════════════════════
# FIGURE 7: Training history (from STATUS.md)
# ═══════════════════════════════════════
# Synthetic chart showing the 3-version evolution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Response length evolution
versions = ['v1\nHandcrafted', 'v2\nHandcrafted\n(eval only)', 'v3\nDeepSeek\nMixed']
base_len = [2083, 2083, 2083]
lora_len = [496, 496, 1776]
x = np.arange(3)
w = 0.3
ax1.bar(x - w/2, base_len, w, color='#E74C3C', label='Original 1.5B (Baseline)', edgecolor='white')
ax1.bar(x + w/2, lora_len, w, color='#2ECC71', label='LoRA Fine-tuned', edgecolor='white')
for i, (b, l) in enumerate(zip(base_len, lora_len)):
    ax1.text(i - w/2, b + 50, str(b), ha='center', fontsize=9)
    ax1.text(i + w/2, l + 50, str(l), ha='center', fontsize=9, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(versions, fontsize=9)
ax1.set_ylabel('Avg Response Length (chars)')
ax1.set_title('Response Length Evolution', fontweight='bold')
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.3, axis='y')

# Right: Eval loss
ax2.bar(['Handcrafted\nLoRA', 'DeepSeek\nMixed LoRA'], [0.5, 0.326], color=['#F39C12', '#2ECC71'],
        width=0.4, edgecolor='white')
ax2.set_ylabel('Eval Loss')
ax2.set_title('Training Eval Loss', fontweight='bold')
for i, v in enumerate([0.5, 0.326]):
    ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

fig.suptitle('Three-Version Experiment Evolution', fontweight='bold', fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig7_training_evolution.png'), dpi=200)
plt.close()
print('Fig7 saved: Training evolution')

print(f'\nAll figures saved to {OUT_DIR}')
