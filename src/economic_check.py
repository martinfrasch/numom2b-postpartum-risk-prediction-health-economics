"""Net monetary benefit of model-guided vs default strategies, using the manuscript's own cost
assumptions (Supplementary Table S2) and recalibrated test-set predictions from revision_pipeline.
A strategy treats patient i when p_i >= C / B, with B = effectiveness x readmission cost; value per
patient = B * TP/n - C * (TP+FP)/n. Writes results/revision/economic_check.json."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.revision_pipeline import OUT, SEED, recalibrate

# base case, Supplementary Table S2: Wen 2020 NRD mean hospital cost per postpartum readmission
# ($6,186 in 2016 USD, weighted same/different hospital) x CPI-U 2016->2024 (1.307, BLS CUUR0000SA0)
# effectiveness base 20%: Lemon 2024 aRR 0.78, Leppin 2014 RR 0.82 (prov:ok literature)
READMIT_COST, EFFECT = 8100, 0.20
B = READMIT_COST * EFFECT                     # value of intervening on one true positive
COSTS = [10, 25, 50, 75, 100, 150, 250, 750]  # $ per flagged patient
te, tr = pd.read_csv(os.path.join(OUT, 'predictions_test.csv')), pd.read_csv(os.path.join(OUT, 'predictions_train_oof.csv'))
te = te.join(pd.read_csv(os.path.join(OUT, 'predictions_test_autogluon.csv')))
tr = tr.join(pd.read_csv(os.path.join(OUT, 'predictions_train_oof_autogluon.csv')))
y, yo = te.pop('y').values, tr.pop('y').values

def value(yy, treat, c):  # $ per patient
    return (B * (treat & (yy == 1)).sum() - c * treat.sum()) / len(yy)

import subprocess
_git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
_dirty = subprocess.run(['git', 'status', '--porcelain', 'src/economic_check.py'], capture_output=True, text=True).stdout.strip()
out = {'script': 'src/economic_check.py', 'git_hash': _git + ('+uncommitted' if _dirty else ''),
       'unit_of_analysis': 'participant', 'predictions_from': 'results/revision/predictions_*.csv (src/revision_pipeline.py)',
       'assumptions': {'readmission_cost': READMIT_COST, 'effectiveness': EFFECT, 'B_per_true_positive': B},
       'prevalence_test': float(y.mean()), 'models': {}}
rng = np.random.RandomState(SEED)
boots = [rng.randint(0, len(y), len(y)) for _ in range(1000)]
for m in te.columns:  # all models
    p = recalibrate(yo, tr[m].values, te[m].values)['platt']
    rows = []
    for c in COSTS:
        t = c / B
        f = lambda idx: (value(y[idx], p[idx] >= t, c),
                         max(value(y[idx], np.ones(len(idx), bool), c), 0.0))  # best default: treat-all or none
        mv, dv = f(np.arange(len(y)))
        gains = [a - b for a, b in (f(i) for i in boots)]
        rows.append({'cost': c, 'threshold_prob': t, 'model_value_per_1000': mv * 1000,
                     'best_default_per_1000': dv * 1000,
                     'best_default': 'treat_all' if value(y, np.ones(len(y), bool), c) > 0 else 'treat_none',
                     'gain_per_1000': (mv - dv) * 1000,
                     'gain_ci95_per_1000': [float(np.percentile(gains, 2.5) * 1000), float(np.percentile(gains, 97.5) * 1000)],
                     'pct_flagged': float((p >= t).mean())})
    out['models'][m] = rows
json.dump(out, open(os.path.join(OUT, 'economic_check.json'), 'w'), indent=2)
for m, rows in list(out['models'].items())[:0]:
    print(m)
    for r in rows:
        print(f"  ${r['cost']:>4} t={r['threshold_prob']*100:5.2f}% model {r['model_value_per_1000']:9.0f} "
              f"default({r['best_default']:10s}) {r['best_default_per_1000']:9.0f} gain {r['gain_per_1000']:7.0f} "
              f"[{r['gain_ci95_per_1000'][0]:.0f}, {r['gain_ci95_per_1000'][1]:.0f}] flagged {r['pct_flagged']*100:5.1f}%")

# Sensitivity to intervention effectiveness (Table S2 range): break-even window per model
EFFECTS = [0.10, 0.20, 0.50]
# Lower readmission-cost bound: Wen 2020 same-hospital median $3,981 (2016 USD) x CPI-U 1.307, a proxy for the
# cheaper readmissions a post-discharge contact can plausibly prevent (Table S2); base-case effectiveness
R_LOW = 5200
SCEN = [('effectiveness_sensitivity', str(eff), READMIT_COST * eff) for eff in EFFECTS] + \
       [('readmission_cost_sensitivity', f'R{R_LOW}_eff{EFFECT}', R_LOW * EFFECT)]
for sec, _, _ in SCEN:
    out[sec] = {}
for m in ['LR_tuned', 'GB_tuned', 'Stacking_orig']:
    p = recalibrate(yo, tr[m].values, te[m].values)['platt']
    for sec, key, b in SCEN:
        rows = []
        for c in range(5, 401, 5):
            treat, all_ = p >= c / b, np.ones(len(y), bool)
            v = lambda tt, idx=np.arange(len(y)): (b * (tt[idx] & (y[idx] == 1)).sum() - c * tt[idx].sum()) / len(idx)
            gains = [v(treat, i) - max(v(all_, i), 0.0) for i in boots[:300]]
            rows.append({'cost': c, 'gain_per_1000': (v(treat) - max(v(all_), 0.0)) * 1000,
                         'ci95': [float(np.percentile(gains, 2.5) * 1000), float(np.percentile(gains, 97.5) * 1000)]})
        pos = [r['cost'] for r in rows if r['gain_per_1000'] > 0]
        sig = [r['cost'] for r in rows if r['ci95'][0] > 0]
        res = {'B_per_true_positive': b, 'cost_break_even_treat_all': float(y.mean() * b),
               'costs_with_positive_gain': pos, 'costs_with_gain_ci_above_zero': sig,
               'max_gain_per_1000': max(r['gain_per_1000'] for r in rows),
               'cost_at_max_gain': max(rows, key=lambda r: r['gain_per_1000'])['cost'], 'rows': rows}
        out[sec].setdefault(m, {})[key] = res
        print(m, key, {k: v_ for k, v_ in res.items() if k != 'rows'})
json.dump(out, open(os.path.join(OUT, 'economic_check.json'), 'w'), indent=2)

# Aggregated risk bins for the interactive page: >=10 test patients per bin, no row-level data
bins = {'note': 'Platt-recalibrated test-set risks grouped into bins of >=10 patients (sorted by risk); '
                'each bin: [lowest risk in bin, events, non-events]. Source: src/economic_check.py',
        'n': int(len(y)), 'events': int(y.sum()), 'models': {}}
for m in ['LR_tuned', 'GB_tuned', 'Stacking_orig', 'LR_orig', 'FLAML_tuned', 'AutoGluon']:
    p = recalibrate(yo, tr[m].values, te[m].values)['platt']
    order = np.argsort(-p); ps, ys = p[order], y[order]
    edges = list(range(0, len(ps), 10))
    rows = []
    for i, s in enumerate(edges):
        e = len(ps) if i == len(edges) - 1 or len(ps) - edges[i + 1] < 10 else edges[i + 1]
        if s >= e: continue
        rows.append([round(float(ps[e - 1]), 6), int(ys[s:e].sum()), int((ys[s:e] == 0).sum())])
        if e == len(ps): break
    assert sum(r[1] + r[2] for r in rows) == len(y) and min(r[1] + r[2] for r in rows) >= 10
    bins['models'][m] = rows
json.dump(bins, open(os.path.join(OUT, 'economic_bins.json'), 'w'))
print('bins written', {k: len(v) for k, v in bins['models'].items()})
