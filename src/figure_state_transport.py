"""Supplementary figure: tuned 18-variable LR's test ROC-AUC transported to each state's first-birth population,
against the state's share of non-Hispanic White first births. EXPLORATORY, POST HOC.
Loads results/expanded/state_reweighted_economics.json and external/wonder/state_first_birth_marginals.json only."""
import ast, json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(os.path.join(ROOT, 'results', 'expanded', 'state_reweighted_economics.json')))
W = json.load(open(os.path.join(ROOT, 'external', 'wonder', 'state_first_birth_marginals.json')))
M = 'LR_tuned_18var'
# state abbreviation -> FIPS, from the producing script's own table (parsed, not imported: importing re-runs it)
src = open(os.path.join(ROOT, 'src', 'state_reweighted_economics.py')).read()
FIPS = {ab: f for f, ab in ast.literal_eval(src[src.index("FIPS = {") + 7: src.index('}', src.index("FIPS = {")) + 1]).items()}


def white_share(ab):
    r = {k: v for k, v in W['race'][FIPS[ab]].items() if k in 'WBHAO'}
    return r['W'] / sum(r.values())


rows = [(white_share(s), v['models'][M]['roc_auc'], *v['models'][M]['roc_auc_ci95'], s)
        for s, v in d['states'].items() if s != 'US' and 'models' in v]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for x, a, lo, hi, s in rows:
    ax.plot([x * 100] * 2, [lo, hi], color='#2a78d6', alpha=0.25, lw=1)
ax.scatter([r[0] * 100 for r in rows], [r[1] for r in rows], s=18, color='#2a78d6', zorder=3, label='State (95% bootstrap CI)')
OFF = {'SD': (-6, -7), 'ND': (6, -7)}   # label placement only
for x, a, lo, hi, s in sorted(rows, key=lambda r: r[1])[:3] + sorted(rows, key=lambda r: r[1])[-1:]:
    ax.annotate(s, (x * 100, a), xytext=OFF.get(s, (5, 0)), textcoords='offset points', fontsize=7, va='center',
                ha='right' if OFF.get(s, (1,))[0] < 0 else 'left')
ax.axhline(d['unweighted']['roc_auc'][M], color='#52514e', ls=':', lw=1, label='nuMoM2b test set, unweighted')
ax.axhline(0.5, color='#898781', lw=1)
ax.set_xlabel('Non-Hispanic White share of first births in the state, 2022-2024 (%)')
ax.set_ylabel('Transported test ROC-AUC')
ax.legend(fontsize=7.5, frameon=False, loc='lower left'); ax.grid(alpha=0.25)
fig.tight_layout()
out = os.path.join(ROOT, 'results', 'manuscript', 'figures', 'figureS1_state_transport.pdf')
fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=200); print('wrote', out)
