"""EXPLORATORY, POST HOC: in which states does post-discharge contact pay for the Medicaid-relevant group, at that
state's own Medicaid contact price?

Combines (1) the government-insured subgroup of the test set, risks Platt-recalibrated on that subgroup's training
OOF (the 'local' calibration of src/subgroup_payer_economics.py, whose split and asserts are imported), with
(2) each state's Medicaid paid amount per claim line from results/expanded/medicaid_contact_costs.json.
One contact = one claim line. Only the two codes billed in >= 40 states are used (home visit, physician phone).

nuMoM2b has no state variable and no public source gives Medicaid postpartum readmission rates by state, so state
prevalence is a sensitivity axis: prior shift (as src/expanded_prevalence_economics.py) to 0.5-2x the subgroup's
observed test prevalence. Readmission cost B is the all-payer base case for every state; decisions depend on c/B,
so a state whose Medicaid readmission payment is a fraction f of B behaves like its contact price divided by f.

Per state, contact type, prevalence and effectiveness, against the best of treat-nobody / treat-all:
  model    : model-targeted contact gain over max(treat-all, 0), bootstrap 95% CI lower bound > 0
  model_near_all : as model, but the model flags >= NEAR_ALL of the group (in practice, contact everyone)
  everyone : treat-all net value CI lower bound > 0 and the model does not beat it
  neither  : otherwise
Writes results/expanded/regional_economics.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.subgroup_payer_economics import te, tr, y, yo, gtr, gte, EXP, EFF, READMIT_COST, MODELS, SEED, recalibrate

PREV_MULT = (0.5, 1.0, 1.5, 2.0)   # design constants
NEAR_ALL = 0.95
CONTACTS = ('home_visit_est_99347_50', 'phone_physician_99441_3')
MCC = json.load(open(os.path.join(EXP, 'medicaid_contact_costs.json')))['groups']

mtr, mte = gtr == 1, gte == 1
yy = y[mte]
assert min(yo[mtr].sum(), yy.sum()) >= 11, 'small cell'
pi0 = yy.mean()
rng = np.random.RandomState(SEED)
boots = np.array([rng.randint(0, len(yy), len(yy)) for _ in range(300)])
odds = lambda p: p / (1 - p)

out = {'_note': 'EXPLORATORY, POST HOC. Source: src/regional_economics.py', 'readmit_cost_base': READMIT_COST,
       'subgroup': 'govt_insured', 'n_test': int(mte.sum()), 'prevalence_test_observed': float(pi0),
       'contacts': CONTACTS, 'near_all': NEAR_ALL, 'models': {}}
for m in MODELS:
    p0 = recalibrate(yo[mtr], tr[m].values[mtr], te[m].values[mte])['platt']
    out['models'][m] = res = {}
    for k in PREV_MULT:
        pi = pi0 * k
        o = odds(p0) * odds(pi) / odds(pi0)
        p, w = o / (1 + o), np.where(yy == 1, pi / pi0, (1 - pi) / (1 - pi0))
        W, Y = w[boots], yy[boots]                      # (300, n) bootstrap weights / outcomes
        for eff in EFF:
            b = READMIT_COST * eff
            for ct in CONTACTS:
                for st, d in MCC[ct]['by_state'].items():
                    c = d['paid_per_line']
                    t = (p >= c / b)[boots]
                    v_all = (W * (b * Y - c)).sum(1) / W.sum(1)
                    v_mod = (W * t * (b * Y - c)).sum(1) / W.sum(1)
                    gain = v_mod - np.maximum(v_all, 0)
                    point_all = (w * (b * yy - c)).sum() / w.sum()
                    point_gain = (w * (p >= c / b) * (b * yy - c)).sum() / w.sum() - max(point_all, 0)
                    g_lo, a_lo = np.percentile(gain, 2.5), np.percentile(v_all, 2.5)
                    flag = float((p >= c / b).mean())
                    verdict = ('model_near_all' if flag >= NEAR_ALL else 'model') if g_lo > 0 else 'everyone' if a_lo > 0 else 'neither'
                    res.setdefault(f'prev_x{k}_eff{eff}', {}).setdefault(ct, {})[st] = {
                        'price': c, 'c_over_B': c / b,
                        'model_gain_per_1000': point_gain * 1000, 'model_gain_ci95': [g_lo * 1000, np.percentile(gain, 97.5) * 1000],
                        'treat_all_per_1000': point_all * 1000, 'treat_all_ci95': [a_lo * 1000, np.percentile(v_all, 97.5) * 1000],
                        'pct_flagged': flag, 'verdict': verdict}
        for eff in EFF:
            r = res[f'prev_x{k}_eff{eff}']
            print(f"{m:30s} prev x{k} eff {eff}: " + '; '.join(
                f"{ct[:5]} model={sorted(s for s, v in r[ct].items() if v['verdict'] == 'model')}"
                f" everyone={sum(v['verdict'] == 'everyone' for v in r[ct].values())}" for ct in CONTACTS))
json.dump(out, open(os.path.join(EXP, 'regional_economics.json'), 'w'), indent=1, default=float)
