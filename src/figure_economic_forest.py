"""Figure: every model's gain over the better no-model strategy (per 1,000 deliveries, base case)
at three contact costs, with 95% bootstrap intervals. Loads results/revision/economic_check.json only."""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'results', 'revision', 'economic_check.json')))['models']
ORDER = [('LR_orig', 'LR, as submitted'), ('LR_orig_noSMOTE', 'LR, as submitted, no SMOTE'), ('LR_tuned', 'LR, tuned'),
         ('RF_orig', 'RF, as submitted'), ('RF_tuned', 'RF, tuned'), ('GB_orig', 'GB, as submitted'), ('GB_tuned', 'GB, tuned'),
         ('Stacking_orig', 'Stacking (submitted bases)'), ('Stacking_tuned', 'Stacking (tuned bases)'),
         ('SoftVote_orig', 'Soft voting (submitted bases)'), ('SoftVote_tuned', 'Soft voting (tuned bases)'),
         ('FLAML_default', 'FLAML, as submitted'), ('FLAML_tuned', 'FLAML, imbalance-aware'), ('AutoGluon', 'AutoGluon')]
GROUP_BREAKS = {3, 7, 11}  # visual gaps between LR / trees / ensembles / AutoML
IMPL = (2300, 4100)        # design assumption (Table S3): running cost per 1,000 deliveries
COSTS = [25, 50, 100]
ys, y = [], 0
for i in range(len(ORDER)):
    if i in GROUP_BREAKS: y += 0.6
    ys.append(y); y += 1
fig, axes = plt.subplots(1, 3, figsize=(11, 5.6), sharey=True)
for ax, c in zip(axes, COSTS):
    ax.axvspan(*IMPL, color='#eb6834', alpha=0.15, lw=0)
    ax.axvline(0, color='#898781', lw=1)
    for (k, _), yy in zip(ORDER, ys):
        r = next(r for r in d[k] if r['cost'] == c)
        g, (lo, hi) = r['gain_per_1000'], r['gain_ci95_per_1000']
        same = lo == hi == g == 0  # recalibrated model reproduces the no-model strategy
        col = '#b9b7b0' if same else ('#2a78d6' if lo > 0 else '#52514e')
        ax.plot([lo, hi], [yy, yy], color=col, lw=2, solid_capstyle='round')
        ax.plot(g, yy, 'o', ms=6, color=col, mec='white', mew=1.2, zorder=3)
    best = next(r for r in d['LR_tuned'] if r['cost'] == c)
    default = 'contact all' if best['best_default'] == 'treat_all' else 'contact none'
    ax.set_title(f'${c} per contact\n(better no-model strategy: {default})', fontsize=10)
    ax.set_xlabel('Gain per 1,000 deliveries'); ax.grid(axis='x', alpha=0.25)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: '$0' if v == 0 else ('−' if v < 0 else '') + f'${abs(v)/1000:g}k'))
axes[0].set_yticks(ys); axes[0].set_yticklabels([n for _, n in ORDER], fontsize=9); axes[0].invert_yaxis()
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend(handles=[Line2D([], [], color='#2a78d6', marker='o', lw=2, label='95% interval above zero'),
                    Line2D([], [], color='#52514e', marker='o', lw=2, label='95% interval includes or is below zero'),
                    Line2D([], [], color='#b9b7b0', marker='o', lw=0, label='Reproduces the no-model strategy (gain = 0)'),
                    Patch(color='#eb6834', alpha=0.25, label="Model's running cost a gain must exceed")],
           loc='lower center', ncol=2, frameon=False, fontsize=8.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
out = os.path.join(ROOT, 'results', 'manuscript', 'figures', 'figure6_economic_all_models.pdf')
fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=200); print('wrote', out)
