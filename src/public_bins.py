"""Coarse risk bins for the public economics explorer (GitHub Pages). Each bin holds >= 11 readmissions and
>= 11 non-readmissions, so no published count falls under the NICHD DASH small-cell threshold. Same format as
results/revision/economic_bins.json but the first element is the bin's MEAN predicted risk (a bin is contacted when its mean
risk reaches the cut-off): [mean risk in bin, events, non-events, mean predicted risk in each of 20 equal slices of the bin (ascending)],
sorted by risk, descending. Each slice averages >= 11 participants' predictions (no individual value, no outcome);
the page uses them to contact part of a bin.
States: state-level summaries only (no per-state bins: 51 re-weightings of the same bins would allow differencing
down to small cells). Reweighted prevalence comes from the raking of src/state_reweighted_economics.py (imported, not
re-implemented); states with < 11 effective readmissions are omitted, as there. The page approximates a state by scaling
the test-set bins' readmission rates to the state's prevalence. Per-state AUC and prevalence intervals are read from
results/expanded/state_reweighted_economics.json; Medicaid contact prices from results/expanded/medicaid_contact_costs.json.
Writes results/revision/economic_bins_public.json."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.revision_pipeline import OUT, recalibrate

MIN_CELL = 11
te, tr = pd.read_csv(os.path.join(OUT, 'predictions_test.csv')), pd.read_csv(os.path.join(OUT, 'predictions_train_oof.csv'))
te = te.join(pd.read_csv(os.path.join(OUT, 'predictions_test_autogluon.csv')))
tr = tr.join(pd.read_csv(os.path.join(OUT, 'predictions_train_oof_autogluon.csv')))
y, yo = te.pop('y').values, tr.pop('y').values

def coarse(p):
    order = np.argsort(-p); ps, ys = p[order], y[order]
    member = np.empty(len(ys), int)
    rows, s = [], 0
    for i in range(len(ys)):
        ev = ys[s:i + 1].sum(); ne = i + 1 - s - ev
        if ev >= MIN_CELL and ne >= MIN_CELL:
            member[order[s:i + 1]] = len(rows); rows.append([round(float(ps[s:i + 1].mean()), 6), int(ev), int(ne)]); s = i + 1
    if s < len(ys):  # remainder (too few events) merges into the last bin
        _, ev, ne = rows.pop(); member[order[s:]] = len(rows); a = int(np.where(member[order] == len(rows))[0][0])
        rows.append([round(float(ps[a:].mean()), 6), ev + int(ys[s:].sum()), ne + int((ys[s:] == 0).sum())])
    for j, r in enumerate(rows):
        sl = np.array_split(np.sort(p[member == j]), 20)
        assert min(len(x) for x in sl) >= MIN_CELL
        r.append([round(float(x.mean()), 5) for x in sl])
    return rows, member

import subprocess
out = {'note': f'Platt-recalibrated test-set risks in bins of >= {MIN_CELL} readmissions and >= {MIN_CELL} non-readmissions '
               '(sorted by risk); each bin: [mean predicted risk in bin, events, non-events, mean risk of 20 equal slices]. Source: src/public_bins.py',
       'script': 'src/public_bins.py', 'git_hash': subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip(),
       'unit_of_analysis': 'participant', 'n': int(len(y)), 'events': int(y.sum()), 'models': {}}
MEMBER = {}
for m in ['LR_tuned', 'GB_tuned', 'Stacking_orig', 'LR_orig', 'FLAML_tuned', 'AutoGluon']:
    rows, MEMBER[m] = coarse(recalibrate(yo, tr[m].values, te[m].values)['platt'])
    assert sum(r[1] + r[2] for r in rows) == len(y) and sum(r[1] for r in rows) == y.sum()
    assert min(min(r[1], r[2]) for r in rows) >= MIN_CELL, (m, rows)
    out['models'][m] = rows

# ---------------------------------------------------------------- states (post hoc covariate-shift transport)
import src.state_reweighted_economics as SR
assert np.array_equal(SR.y, y), 'state script test rows differ from revision predictions'
EXP = os.path.join(os.path.dirname(OUT), 'expanded')
SJ = json.load(open(os.path.join(EXP, 'state_reweighted_economics.json')))['states']
MCC = json.load(open(os.path.join(EXP, 'medicaid_contact_costs.json')))['groups']
out['states'], out['states_note'] = {}, ('POST HOC, EXPLORATORY. Test set raked to each state\'s first births (CDC WONDER 2022-24) on race/ethnicity, '
    'Medicaid, education, BMI and age; assumes risk given these is the same as in nuMoM2b. State-level summaries only. '
    'Source: src/public_bins.py with src/state_reweighted_economics.py')
for st in ['US'] + [SR.FIPS[f] for f in SR.STATES]:
    w = SR.rake(SR.X, SR.TGT[st])
    ess, ess_ev = w.sum() ** 2 / (w ** 2).sum(), w[y == 1].sum() ** 2 / (w[y == 1] ** 2).sum()
    if ess_ev < SR.MIN_EVENT_ESS:
        assert 'suppressed' in SJ[st]; out['states'][st] = {'suppressed': True}; continue
    w = w * len(y) / w.sum()
    a = SJ[st]['models']['LR_tuned_18var']
    price = {k: (MCC[g]['by_state'].get(st, {}).get('paid_per_line'), MCC[g]['national_paid_per_line'])
             for k, g in (('home', 'home_visit_est_99347_50'), ('phone', 'phone_physician_99441_3'))}
    out['states'][st] = {
        'births': SR.births.get(st, sum(SR.births.values())), 'medicaid_share': round(float(SR.TGT[st][1][0]), 4),
        'ess': int(round(float(ess), -1)),
        'prevalence': round(float(np.average(y, weights=w)), 4), 'prevalence_ci95': [round(v, 4) for v in SJ[st]['prevalence_ci95']],
        'auc_lr': round(a['roc_auc'], 3), 'auc_lr_ci95': [round(v, 3) for v in a['roc_auc_ci95']],
        'price_home': round(price['home'][0] or price['home'][1], 2), 'price_home_is_state': price['home'][0] is not None,
        'price_phone': round(price['phone'][0] or price['phone'][1], 2), 'price_phone_is_state': price['phone'][0] is not None}
    assert abs(out['states'][st]['prevalence'] - SJ[st]['prevalence']) < 1e-4, st
    assert 'models' not in out['states'][st]
json.dump(out, open(os.path.join(OUT, 'economic_bins_public.json'), 'w'), indent=1)
print(len([s for s in out['states'].values() if 'prevalence' in s]), 'populations (states + US);', [k for k, v in out['states'].items() if 'suppressed' in v], 'suppressed')
print({m: [r[1:3] for r in v] for m, v in out['models'].items()})
