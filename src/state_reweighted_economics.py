"""EXPLORATORY, POST HOC: state-transported performance and economics by covariate-shift reweighting.

The nuMoM2b test set (participants; split and disjointness imported from src/subgroup_payer_economics.py and
re-asserted here) is raked (iterative proportional fitting) to each state's first-birth population: singleton,
live-birth-order-1 births 2022-2024, CDC WONDER Natality expanded (external/wonder/state_first_birth_marginals.json).
Raking margins, all recorded on both the nuMoM2b form and the US birth certificate:
  race/ethnicity (CRace, 5 groups), government insurance vs Medicaid at delivery (Ins_Govt), education (6 groups),
  pre-pregnancy vs visit-1 BMI category (5 groups), age group (4 groups).
Smoking is not raked: the certificate records cigarettes only and under-reports, nuMoM2b records any tobacco.
A row missing a margin is not adjusted on that margin; target categories absent from a (bootstrap) sample are dropped
and the rest renormalised. Raking is refitted inside every bootstrap resample.

Assumes covariate shift: readmission risk given these covariates is the same in every state as in nuMoM2b's 8 centres.
Risks are Platt-recalibrated on all training OOF (as src/expanded_economics.py) and not re-fitted per state.

Economics per woman i: contact if p_i >= c_i / B_i, B_i = effectiveness x readmission cost; gain vs the best default
(contact everyone or no one) chosen separately within each payer group. Government-insured women: c = the state's
Medicaid paid amount per line (national when the state has < 1,000 lines), B base $8,100 (Supplementary Table S2). Other women, scenarios:
  same       : c and B as Medicaid
  billed     : commercial fee-for-service, c x 1.88 / 0.66, B x 2.55
  programme  : contact cost not price-scaled (salaried programme), c as Medicaid, B x 2.55
Writes results/expanded/state_reweighted_economics.json.
"""
import json, os, sys, subprocess
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.subgroup_payer_economics import df, itr, ite, te, tr, y, yo, EXP, MODELS, SEED, READMIT_COST, recalibrate, RAW

EFF = (0.10, 0.20, 0.50)        # Supplementary Table S2: base 20%, range 10-50%
N_BOOT = 300
MIN_EVENT_ESS = 11              # design constant: suppress states whose effective number of events is < 11
NEAR_ALL = 0.95
# literature ratios (prov:ok): RAND RR-A1144-2 (2024) commercial/Medicare inpatient 2.55, professional 1.88;
# KFF/Urban Medicaid-to-Medicare fee index 2024, primary care 0.66; Medicaid inpatient taken at Medicare parity (MACPAC).
SCEN = {'same': (1.0, 1.0), 'billed': (1.88 / 0.66, 2.55), 'programme': (1.0, 2.55)}   # (contact x, readmission x)
CONTACTS = ('home_visit_est_99347_50', 'phone_physician_99441_3')
FIPS = {'01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE', '11': 'DC',
        '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA', '20': 'KS', '21': 'KY',
        '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN', '28': 'MS', '29': 'MO', '30': 'MT',
        '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM', '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH',
        '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI', '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT',
        '50': 'VT', '51': 'VA', '53': 'WA', '54': 'WV', '55': 'WI', '56': 'WY'}

# ---------------------------------------------------------------- covariates of the test participants (codes 1..K, -1 missing)
ids = df.PublicID.values
assert not set(ids[itr]) & set(ids[ite]), 'unit leak'  # split-unit:ok one row per participant
cov = pd.read_csv(RAW, usecols=['PublicID', 'CRace', 'Ins_Govt', 'Education', 'BMI_Cat', 'AgeCat_V1'], low_memory=False)
cov = cov.set_index('PublicID').loc[ids[ite]]
X = np.column_stack([pd.to_numeric(cov[c], errors='coerce').fillna(0).astype(int).values - 1
                     for c in ('CRace', 'Ins_Govt', 'Education', 'BMI_Cat', 'AgeCat_V1')])   # -1 = missing
K = (5, 2, 6, 5, 4)
govt = X[:, 1] == 0

# ---------------------------------------------------------------- state targets from WONDER (unknowns dropped)
W = json.load(open('external/wonder/state_first_birth_marginals.json'))
EDU = {'8th grade or less': 0, '9th through 12th grade with no diploma': 0, 'High school graduate or GED completed': 1,
       'Some college credit, but not a degree': 2, 'Associate degree (AA, AS)': 3, "Bachelor's degree (BA, AB, BS)": 4}
BMI = {'Underweight <18.5': 0, 'Normal 18.5-24.9': 1, 'Overweight 25.0-29.9': 2, 'Obesity I 30.0-34.9': 3,
       'Obesity II 35.0-39.9': 4, 'Extreme Obesity III > 39.9': 4}
def vec(d, f, k):
    v = np.zeros(k)
    for lab, n in d.items():
        j = f(lab)
        if j is not None: v[j] += n
    return v / v.sum()
def targets(st):
    return [vec(W['race'][st], lambda l: 'WBHAO'.index(l), 5),
            vec(W['payer'][st], lambda l: 0 if l == 'Medicaid' else None if 'Unknown' in l else 1, 2),
            vec(W['educ'][st], lambda l: EDU.get(l, None if 'Unknown' in l or l in ('Excluded', 'Not Reported') else 5), 6),
            vec(W['bmi'][st], lambda l: BMI.get(l), 5),
            vec(W['age'][st], lambda l: ['<18', '18-34', '35-39', '40+'].index(l), 4)]
STATES = sorted(W['payer'], key=lambda s: FIPS[s])
TGT = {FIPS[s]: targets(s) for s in STATES}
births = {FIPS[s]: int(sum(W['age'][s].values())) for s in STATES}
nat = [sum(TGT[FIPS[s]][j] * births[FIPS[s]] for s in STATES) / sum(births.values()) for j in range(5)]
TGT['US'] = nat


def rake(Xs, tgt, iters=200, tol=1e-8):
    w = np.ones(len(Xs))
    for _ in range(iters):
        worst = 0.0
        for j, t in enumerate(tgt):
            obs = Xs[:, j] >= 0
            present = np.bincount(Xs[obs, j], minlength=len(t)) > 0
            tt = np.where(present, t, 0); tt = tt / tt.sum()
            cur = np.bincount(Xs[obs, j], weights=w[obs], minlength=len(t)) / w[obs].sum()
            f = np.divide(tt, cur, out=np.ones_like(tt), where=cur > 0)
            w[obs] *= f[Xs[obs, j]]
            worst = max(worst, np.abs(cur - tt)[present].max())
        if worst < tol: break
    return w / w.mean()


def econ(p, yy, w, c, b, g):
    """value per woman: model policy and best per-payer-group default; plus fraction flagged."""
    net = b * yy - c
    treat = p >= c / b
    vm = (w * treat * net).sum() / w.sum()
    vd = sum(max((w * net)[g == k].sum(), 0.0) for k in (0, 1)) / w.sum()
    return vm - vd, vd, float((w * treat).sum() / w.sum())


if __name__ == '__main__':   # importable: X, TGT, rake, STATES, FIPS, births for src/public_bins.py
    MCC = json.load(open(os.path.join(EXP, 'medicaid_contact_costs.json')))['groups']
    P = {m: recalibrate(yo, tr[m].values, te[m].values)['platt'] for m in MODELS}
    rng = np.random.RandomState(SEED)
    boots = [rng.randint(0, len(y), len(y)) for _ in range(N_BOOT)]
    git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip() or 'uncommitted'
    if subprocess.run(['git', 'status', '--porcelain', __file__], capture_output=True, text=True).stdout.strip():
        git += '+uncommitted'
    out = {'_note': 'EXPLORATORY, POST HOC. Covariate-shift transport of the test set to state first-birth populations.',
           'script': 'src/state_reweighted_economics.py', 'git_hash': git, 'unit_of_analysis': 'participant',
           'splitter': 'train_test_split (imported from src/subgroup_payer_economics.py)', 'groups_passed': False,
           'n_train_units': int(len(itr)), 'n_test_units': int(len(ite)), 'n_overlap_units': 0,
           'n_boot': N_BOOT, 'effectiveness': EFF, 'scenarios': {k: {'contact_x': v[0], 'readmission_x': v[1]} for k, v in SCEN.items()},
           'readmit_cost_base': READMIT_COST, 'min_event_ess': MIN_EVENT_ESS,
           'unweighted': {'prevalence': float(y.mean()), 'roc_auc': {m: float(roc_auc_score(y, te[m].values)) for m in MODELS}},
           'states': {}}
    for st in ['US'] + [FIPS[s] for s in STATES]:
        tgt = TGT[st]
        w0 = rake(X, tgt)
        for j, t in enumerate(tgt):   # raking reproduces every margin
            obs = X[:, j] >= 0
            assert np.allclose(np.bincount(X[obs, j], weights=w0[obs], minlength=len(t)) / w0[obs].sum(), t, atol=1e-4), (st, j)
        ess, ess_ev = w0.sum() ** 2 / (w0 ** 2).sum(), w0[y == 1].sum() ** 2 / (w0[y == 1] ** 2).sum()
        rec = {'births_first_singleton_2022_24': births.get(st, sum(births.values())), 'ess': float(ess), 'event_ess': float(ess_ev),
               'max_weight': float(w0.max())}
        if ess_ev < MIN_EVENT_ESS:
            rec['suppressed'] = f'effective events {ess_ev:.1f} < {MIN_EVENT_ESS}'
            out['states'][st] = rec; print(st, rec['suppressed']); continue
        W_b = [rake(X[i], tgt) for i in boots]
        prev_b = [np.average(y[i], weights=wb) for i, wb in zip(boots, W_b)]
        rec['prevalence'] = float(np.average(y, weights=w0))
        rec['prevalence_ci95'] = [float(np.percentile(prev_b, 2.5)), float(np.percentile(prev_b, 97.5))]
        rec['models'] = {}
        for m in MODELS:
            p = P[m]
            auc_b = [roc_auc_score(y[i], p[i], sample_weight=wb) for i, wb in zip(boots, W_b) if 0 < y[i].sum() < len(i)]
            r = {'roc_auc': float(roc_auc_score(y, p, sample_weight=w0)),
                 'roc_auc_ci95': [float(np.percentile(auc_b, 2.5)), float(np.percentile(auc_b, 97.5))], 'econ': {}}
            for ct in CONTACTS:
                cm = MCC[ct]['by_state'].get(st, {}).get('paid_per_line', MCC[ct]['national_paid_per_line'])
                for sc, (kc, kb) in SCEN.items():
                    for e in EFF:
                        c = np.where(govt, cm, cm * kc)
                        b = np.where(govt, READMIT_COST * e, READMIT_COST * e * kb)
                        g = (~govt).astype(int)
                        gain, vd, flag = econ(p, y, w0, c, b, g)
                        gb = [econ(p[i], y[i], wb, c[i], b[i], g[i]) for i, wb in zip(boots, W_b)]
                        glo, ghi = np.percentile([x[0] for x in gb], [2.5, 97.5]); dlo = np.percentile([x[1] for x in gb], 2.5)
                        r['econ'][f'{ct}|{sc}|{e}'] = {
                            'medicaid_price': cm, 'price_is_state': st in MCC[ct]['by_state'],
                            'gain_per_1000': gain * 1000, 'gain_ci95': [glo * 1000, ghi * 1000],
                            'default_per_1000': vd * 1000, 'default_ci95_lo': dlo * 1000, 'pct_flagged': flag,
                            'verdict': ('model_near_all' if flag >= NEAR_ALL else 'model') if glo > 0 else 'default' if dlo > 0 else 'neither'}
            rec['models'][m] = r
        out['states'][st] = rec
        r = rec['models']['LR_tuned_18var']
        print(f"{st}: ESS {ess:.0f} (events {ess_ev:.1f}), prev {rec['prevalence']:.4f}, AUC {r['roc_auc']:.3f} "
              f"[{r['roc_auc_ci95'][0]:.3f}, {r['roc_auc_ci95'][1]:.3f}]; home|same|0.2 {r['econ']['home_visit_est_99347_50|same|0.2']['verdict']}, "
              f"home|programme|0.2 {r['econ']['home_visit_est_99347_50|programme|0.2']['verdict']}")
    # ---------------------------------------------------------------- cross-state summary (read by src/expanded_report.py)
    S = {k: v for k, v in out['states'].items() if k != 'US' and 'models' in v}
    share = {'nh_white': (0, 0), 'nh_black': (0, 1), 'hispanic': (0, 2), 'medicaid': (1, 0)}
    out['summary'] = {'n_states_reported': len(S), 'suppressed': sorted(k for k, v in out['states'].items() if 'suppressed' in v),
                      'prevalence_min_median_max': [float(f(np.array([v['prevalence'] for v in S.values()]))) for f in (np.min, np.median, np.max)],
                      'ess_min_max': [min(v['ess'] for v in S.values()), max(v['ess'] for v in S.values())],
                      'max_weight': max(v['max_weight'] for v in out['states'].values()), 'models': {}}
    for m in MODELS:
        auc = np.array([S[s]['models'][m]['roc_auc'] for s in S])
        out['summary']['models'][m] = {
            'roc_auc_min_median_max': [float(auc.min()), float(np.median(auc)), float(auc.max())],
            'n_states_auc_ci_above_0.5': int(sum(S[s]['models'][m]['roc_auc_ci95'][0] > 0.5 for s in S)),
            'corr_auc_with_state_share': {k: float(np.corrcoef([TGT[s][j][c] for s in S], auc)[0, 1]) for k, (j, c) in share.items()},
            'model_verdict_count_by_state': {s: sum(x['verdict'] == 'model' for x in S[s]['models'][m]['econ'].values()) for s in S},
            'verdict_counts': {k: dict(pd.Series([S[s]['models'][m]['econ'][k]['verdict'] for s in S]).value_counts())
                               for k in S[next(iter(S))]['models'][m]['econ']}}
    out['summary']['corr_prevalence_with_medicaid_share'] = float(np.corrcoef([TGT[s][1][0] for s in S], [S[s]['prevalence'] for s in S])[0, 1])
    json.dump(out, open(os.path.join(EXP, 'state_reweighted_economics.json'), 'w'), indent=1, default=lambda o: int(o) if isinstance(o, np.integer) else float(o))
