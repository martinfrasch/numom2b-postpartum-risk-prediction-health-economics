"""EXPLORATORY, POST HOC: does the economic window survive regional differences in readmission prevalence?

Regional price levels that scale contact cost c and readmission cost B together do not change the decision
(treat when risk >= c/B; gain / B depends on c/B only), so what regional variation can move is prevalence,
the c/B ratio and effectiveness. Prevalence shift is simulated by prior shift (label shift):
  - Platt-recalibrated risks (fitted on training OOF, as src/expanded_economics.py) get their odds multiplied by
    odds(pi') / odds(pi_test) (so x1.0 reproduces economics.json);
  - test outcomes are reweighted (positives pi'/pi_test, negatives (1-pi')/(1-pi_test)), bootstrap as before.
Assumes the case mix given readmission/no readmission is unchanged across regions (label-shift assumption).
Prevalence multipliers and the cost grid are design constants. Writes results/expanded/prevalence_economics.json.
"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.revision_pipeline import OUT as REV_OUT, SEED, recalibrate

READMIT_COST = 8100                       # base case, Supplementary Table S2 (as src/economic_check.py)
PREV_MULT = (0.5, 1.0, 1.5, 2.0)           # design constants
EFF = (0.20, 0.50)
MODELS = ('LR_tuned_18var', 'tier2_to_discharge__L1_LR', 'tier2_to_discharge__RFE_HGB', 'AutoGluon_tier2_stab')
EXP = os.path.join(os.path.dirname(REV_OUT), 'expanded')

te, tr = pd.read_csv(os.path.join(EXP, 'predictions_test.csv')), pd.read_csv(os.path.join(EXP, 'predictions_train_oof.csv'))
rte, rtr = pd.read_csv(os.path.join(REV_OUT, 'predictions_test.csv')), pd.read_csv(os.path.join(REV_OUT, 'predictions_train_oof.csv'))
assert np.array_equal(te.y.values, rte.y.values) and np.array_equal(tr.y.values, rtr.y.values), 'split mismatch'
te['LR_tuned_18var'], tr['LR_tuned_18var'] = rte.LR_tuned, rtr.LR_tuned
ag = os.path.join(EXP, 'predictions_test_autogluon.csv')
if os.path.exists(ag):
    te = te.join(pd.read_csv(ag)); tr = tr.join(pd.read_csv(os.path.join(EXP, 'predictions_train_oof_autogluon.csv')))
y, yo = te.y.values, tr.y.values
pi_tr, pi_te = yo.mean(), y.mean()
rng = np.random.RandomState(SEED)
boots = [rng.randint(0, len(y), len(y)) for _ in range(300)]
odds = lambda p: p / (1 - p)

out = {'_note': 'EXPLORATORY, POST HOC. Prior-shift sensitivity. Source: src/expanded_prevalence_economics.py',
       'prevalence_test_observed': float(pi_te), 'prevalence_train': float(pi_tr), 'readmit_cost_base': READMIT_COST,
       'models': {}}
for m in [c for c in MODELS if c in te]:
    p0 = recalibrate(yo, tr[m].values, te[m].values)['platt']
    out['models'][m] = {}
    for k in PREV_MULT:
        pi = pi_te * k
        o = odds(p0) * odds(pi) / odds(pi_te)  # x1.0 reproduces economics.json exactly
        p = o / (1 + o)
        w = np.where(y == 1, pi / pi_te, (1 - pi) / (1 - pi_te))
        for eff in EFF:
            b = READMIT_COST * eff
            rows = []
            for c in range(5, 401, 5):
                treat = p >= c / b
                def v(tt, i=np.arange(len(y))):
                    return (w[i] * (b * (tt[i] & (y[i] == 1)) - c * tt[i])).sum() / w[i].sum()
                gains = [v(treat, i) - max(v(np.ones(len(y), bool), i), 0.0) for i in boots]
                rows.append({'cost': c, 'gain_per_1000': (v(treat) - max(v(np.ones(len(y), bool)), 0.0)) * 1000,
                             'ci95': [float(np.percentile(gains, 2.5) * 1000), float(np.percentile(gains, 97.5) * 1000)]})
            best = max(rows, key=lambda r: r['gain_per_1000'])
            ci0 = [r['cost'] for r in rows if r['ci95'][0] > 0]
            out['models'][m][f'prev_x{k}_eff{eff}'] = {
                'prevalence': float(pi), 'effectiveness': eff, 'costs_with_gain_ci_above_zero': ci0,
                'window_c_over_B': [ci0[0] / b, ci0[-1] / b] if ci0 else None,
                'max_gain_per_1000': best['gain_per_1000'], 'cost_at_max_gain': best['cost'], 'ci95_at_max_gain': best['ci95']}
            print(f"{m:32s} prev {pi:.4f} eff {eff}: CI>0 at {ci0[:1] + ci0[-1:] if ci0 else 'none'}; "
                  f"max {best['gain_per_1000']:.0f} at ${best['cost']} [{best['ci95'][0]:.0f}, {best['ci95'][1]:.0f}]")
json.dump(out, open(os.path.join(EXP, 'prevalence_economics.json'), 'w'), indent=1)
