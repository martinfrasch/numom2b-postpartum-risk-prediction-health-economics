"""Figure: gain of the tuned LR over the better no-model strategy vs cost per contact, by effectiveness.
Loads results/revision/economic_check.json only (no literals for results)."""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'results', 'revision', 'economic_check.json')))
S = d['effectiveness_sensitivity']['LR_tuned']
IMPL = (2300, 4100)  # design assumption: annualised running cost per 1,000 deliveries (Table S3)
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, (eff, r) in zip(axes, sorted(S.items(), key=lambda kv: float(kv[0]))):
    c = [x['cost'] for x in r['rows']]; g = [x['gain_per_1000'] for x in r['rows']]
    lo = [x['ci95'][0] for x in r['rows']]; hi = [x['ci95'][1] for x in r['rows']]
    ax.fill_between(c, lo, hi, color='#2a78d6', alpha=0.15, lw=0, label='95% bootstrap interval')
    ax.plot(c, g, color='#2a78d6', lw=2, label='Gain of tuned LR')
    ax.axhspan(*IMPL, color='#eb6834', alpha=0.15, lw=0, label='Model running cost to recover')
    ax.axhline(0, color='#898781', lw=1)
    ax.axvline(r['cost_break_even_treat_all'], color='#52514e', lw=1, ls=':', label='Contact-all = contact-none')
    ax.set_title(f'{float(eff)*100:.0f}% of reached readmissions prevented', fontsize=10)
    ax.set_xlabel('Cost per patient contacted ($)'); ax.grid(alpha=0.25); ax.set_xlim(0, 200)
axes[0].set_ylabel('Gain per 1,000 deliveries ($)')
axes[0].legend(fontsize=7.5, loc='upper right', frameon=False)
fig.tight_layout()
out = os.path.join(ROOT, 'results', 'manuscript', 'figures', 'figure5_economic_window.pdf')
fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=200); print('wrote', out)
