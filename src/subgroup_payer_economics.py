"""EXPLORATORY, POST HOC: economic window by payer -- government-insured (Ins_Govt = 1, the Medicaid-relevant group)
vs not (Ins_Govt = 2). nuMoM2b has no site/region variable, so payer is the link to Medicaid cost data.

Same models, split and economics as src/expanded_economics.py, restricted to each subgroup of the test set.
Two calibrations: 'overall' (Platt fitted on all training OOF, i.e. one model deployed everywhere) and
'local' (Platt refitted on the subgroup's training OOF, i.e. recalibrated for a Medicaid deployment).
Subgroups with < 11 events in train or test are not reported (small-cell rule). Writes results/expanded/payer_economics.json.
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.revision_pipeline import OUT as REV_OUT, SEED, recalibrate
from src.expanded_features import RAW

READMIT_COST = 8100                     # base case, Supplementary Table S2 (as src/economic_check.py)
EFF = (0.20, 0.50)
MODELS = ('LR_tuned_18var', 'tier2_to_discharge__L1_LR', 'tier2_to_discharge__RFE_HGB', 'AutoGluon_tier2_stab')
EXP = os.path.join(os.path.dirname(REV_OUT), 'expanded')

# payer per participant, in the main analysis' row order and split (asserted)
df = pd.read_csv(RAW, usecols=['PublicID', 'CMAJ01', 'Ins_Govt'], low_memory=False)
df = df[df.CMAJ01.notna()].reset_index(drop=True)
y_all = (pd.to_numeric(df.CMAJ01, errors='coerce') == 1).astype(int).values
itr, ite = train_test_split(np.arange(len(y_all)), test_size=0.30, random_state=SEED, stratify=y_all)
assert not set(df.PublicID.values[itr]) & set(df.PublicID.values[ite]), 'participant overlap'
g = pd.to_numeric(df.Ins_Govt, errors='coerce').values
gtr, gte = g[itr], g[ite]

te, tr = pd.read_csv(os.path.join(EXP, 'predictions_test.csv')), pd.read_csv(os.path.join(EXP, 'predictions_train_oof.csv'))
rte, rtr = pd.read_csv(os.path.join(REV_OUT, 'predictions_test.csv')), pd.read_csv(os.path.join(REV_OUT, 'predictions_train_oof.csv'))
assert np.array_equal(te.y.values, y_all[ite]) and np.array_equal(tr.y.values, y_all[itr]), 'split mismatch'
te['LR_tuned_18var'], tr['LR_tuned_18var'] = rte.LR_tuned, rtr.LR_tuned
te = te.join(pd.read_csv(os.path.join(EXP, 'predictions_test_autogluon.csv')))
tr = tr.join(pd.read_csv(os.path.join(EXP, 'predictions_train_oof_autogluon.csv')))
y, yo = te.y.values, tr.y.values
rng = np.random.RandomState(SEED)


def window(p, yy, boots):
    res = {}
    for eff in EFF:
        b, rows = READMIT_COST * eff, []
        for c in range(5, 401, 5):
            treat, all_ = p >= c / b, np.ones(len(yy), bool)
            v = lambda tt, i=np.arange(len(yy)): (b * (tt[i] & (yy[i] == 1)).sum() - c * tt[i].sum()) / len(i)
            gains = [v(treat, i) - max(v(all_, i), 0.0) for i in boots]
            rows.append({'cost': c, 'gain_per_1000': (v(treat) - max(v(all_), 0.0)) * 1000,
                         'ci95': [float(np.percentile(gains, 2.5) * 1000), float(np.percentile(gains, 97.5) * 1000)],
                         'pct_flagged': float(treat.mean())})
        best = max(rows, key=lambda r: r['gain_per_1000'])
        res[str(eff)] = {'costs_with_gain_ci_above_zero': [r['cost'] for r in rows if r['ci95'][0] > 0],
                         'max_gain_per_1000': best['gain_per_1000'], 'cost_at_max_gain': best['cost'],
                         'ci95_at_max_gain': best['ci95'], 'rows': rows}
    return res


if __name__ == '__main__':
    out = {'_note': 'EXPLORATORY, POST HOC. Source: src/subgroup_payer_economics.py', 'readmit_cost_base': READMIT_COST,
           'subgroups': {}}
    for name, code in (('govt_insured', 1), ('not_govt_insured', 2)):
        mtr, mte = gtr == code, gte == code
        etr, ete = int(yo[mtr].sum()), int(y[mte].sum())
        if min(etr, ete) < 11:
            out['subgroups'][name] = {'suppressed': 'fewer than 11 events'}
            continue
        yy = y[mte]
        boots = [rng.randint(0, mte.sum(), mte.sum()) for _ in range(300)]
        sg = {'n_train': int(mtr.sum()), 'n_test': int(mte.sum()), 'events_train': etr, 'events_test': ete,
              'prevalence_train': float(yo[mtr].mean()), 'prevalence_test': float(yy.mean()), 'models': {}}
        for m in MODELS:
            p_over = recalibrate(yo, tr[m].values, te[m].values)['platt'][mte]
            p_loc = recalibrate(yo[mtr], tr[m].values[mtr], te[m].values[mte])['platt']
            auc_b = [roc_auc_score(yy[i], te[m].values[mte][i]) for i in boots if 0 < yy[i].sum() < len(i)]
            sg['models'][m] = {'roc_auc': float(roc_auc_score(yy, te[m].values[mte])),
                               'roc_auc_ci95': [float(np.percentile(auc_b, 2.5)), float(np.percentile(auc_b, 97.5))],
                               'overall_calibration': window(p_over, yy, boots), 'local_calibration': window(p_loc, yy, boots)}
            for cal in ('overall_calibration', 'local_calibration'):
                for eff in EFF:
                    r = sg['models'][m][cal][str(eff)]
                    print(f"{name:16s} {m:30s} {cal[:5]} eff {eff}: AUC {sg['models'][m]['roc_auc']:.3f}; "
                          f"CI>0 at {r['costs_with_gain_ci_above_zero']}; max {r['max_gain_per_1000']:.0f} at ${r['cost_at_max_gain']}")
        out['subgroups'][name] = sg
    json.dump(out, open(os.path.join(EXP, 'payer_economics.json'), 'w'), indent=1)
