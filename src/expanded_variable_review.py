"""Training-only univariate AUC + full codebook labels for the variables selected in results/expanded/expanded.json.
No cell counts are written. Usage: python src/expanded_variable_review.py results/expanded/selected_variables_review.json"""
import ast, json, sys, numpy as np, pandas as pd
sys.path.insert(0, '.'); sys.path.insert(0, 'src')
from expanded_features import inventory, load
from sklearn.metrics import roc_auc_score
e = json.load(open('results/expanded/expanded.json'))['tiers']
sel = {}
def add(t, m, key, rows):
    for r in rows:
        v = r['var']; sel.setdefault(v, []).append(f"{t.split('_')[0]}:{m}" + (f" {r['freq']:.2f}" if 'freq' in r else ''))
for t in ('tier1_antenatal', 'tier2_to_discharge'):
    add(t, 'L1stab', 'stability_top30', [r for r in e[t]['L1_LR']['stability_top30'] if r['freq'] >= 0.3])
    for m in ('RFE_LR', 'RFE_HGB'):
        r = e[t][m]['rfe_selected']; add(t, m, 'rfe', ast.literal_eval(r) if isinstance(r, str) else r)
tiers, labels, forms = inventory()
X, y, itr, ite, ref = load(tiers)
out = []
for v, where in sel.items():
    base = v.replace('__missing', '')
    x = X.iloc[itr][base]
    if v.endswith('__missing'): xv = x.isna().astype(float)
    else: xv = x.fillna(x.median())
    ok = xv.notna()
    auc = roc_auc_score(y[itr][ok.values], xv[ok].values) if xv[ok].nunique() > 1 else float('nan')
    out.append({'var': v, 'label': labels.get(base, ''), 'selected_by': where,
                'pct_missing_train': float(x.isna().mean()), 'auc_train': float(auc), 'n_unique': int(x.nunique())})
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(len(out))
