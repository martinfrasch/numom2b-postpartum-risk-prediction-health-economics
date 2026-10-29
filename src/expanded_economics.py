"""EXPLORATORY, POST HOC: economic window for the expanded-predictor models (src/expanded_features.py),
computed exactly as in src/economic_check.py (effectiveness_sensitivity): Platt recalibration fitted
on training OOF predictions, treat when risk >= c/B, gain over the better of treat-all / treat-none,
bootstrap 95% interval. The 18-variable tuned LR is included as the reference.
Writes results/expanded/economics.json."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.revision_pipeline import OUT as REV_OUT, SEED, recalibrate

READMIT_COST = 8100         # base case, Supplementary Table S2 (same constant as src/economic_check.py)
EXP = os.path.join(os.path.dirname(REV_OUT), 'expanded')
RUNNING_COST = (2300, 4100)  # implementation cost per 1,000 deliveries, Supplementary Table S2
te, tr = pd.read_csv(os.path.join(EXP, 'predictions_test.csv')), pd.read_csv(os.path.join(EXP, 'predictions_train_oof.csv'))
rte, rtr = pd.read_csv(os.path.join(REV_OUT, 'predictions_test.csv')), pd.read_csv(os.path.join(REV_OUT, 'predictions_train_oof.csv'))
assert np.array_equal(te.y.values, rte.y.values) and np.array_equal(tr.y.values, rtr.y.values), 'split mismatch'
te['LR_tuned_18var'], tr['LR_tuned_18var'] = rte.LR_tuned, rtr.LR_tuned
y, yo = te.pop('y').values, tr.pop('y').values
rng = np.random.RandomState(SEED)
boots = [rng.randint(0, len(y), len(y)) for _ in range(300)]  # as economic_check effectiveness_sensitivity

out = {'_note': 'EXPLORATORY, POST HOC. Source: src/expanded_economics.py', 'running_cost_per_1000': RUNNING_COST,
       'prevalence_test': float(y.mean()), 'models': {}}
for m in te.columns:
    p = recalibrate(yo, tr[m].values, te[m].values)['platt']
    res = {}
    for eff in (0.10, 0.20, 0.50):
        b = READMIT_COST * eff
        rows = []
        for c in range(5, 401, 5):
            treat, all_ = p >= c / b, np.ones(len(y), bool)
            v = lambda tt, idx=np.arange(len(y)): (b * (tt[idx] & (y[idx] == 1)).sum() - c * tt[idx].sum()) / len(idx)
            gains = [v(treat, i) - max(v(all_, i), 0.0) for i in boots]
            rows.append({'cost': c, 'gain_per_1000': (v(treat) - max(v(all_), 0.0)) * 1000,
                         'ci95': [float(np.percentile(gains, 2.5) * 1000), float(np.percentile(gains, 97.5) * 1000)],
                         'pct_flagged': float(treat.mean())})
        best = max(rows, key=lambda r: r['gain_per_1000'])
        res[str(eff)] = {'cost_break_even_treat_all': float(y.mean() * b),
                         'costs_with_gain_ci_above_zero': [r['cost'] for r in rows if r['ci95'][0] > 0],
                         'costs_with_gain_ci_above_running_cost_low': [r['cost'] for r in rows if r['ci95'][0] > RUNNING_COST[0]],
                         'max_gain_per_1000': best['gain_per_1000'], 'cost_at_max_gain': best['cost'],
                         'ci95_at_max_gain': best['ci95'], 'rows': rows}
        print(f"{m:40s} eff {eff:.2f}: max gain {best['gain_per_1000']:8.0f} at ${best['cost']} "
              f"[{best['ci95'][0]:.0f}, {best['ci95'][1]:.0f}]; CI>0 at {res[str(eff)]['costs_with_gain_ci_above_zero']}")
    out['models'][m] = res
json.dump(out, open(os.path.join(EXP, 'economics.json'), 'w'), indent=1)
