"""EXPLORATORY, POST HOC: algorithmic, time-anchored expansion of the predictor set.

Question: does data the cohort already holds -- beyond the 18 predefined sociodemographic
variables -- improve prediction of 14-day postpartum readmission, and does it widen the
economic window in which a model can pay?

Design (fixed before any model was fitted):
  unit of analysis = participant (PublicID); the main paper's 70/30 split (seed 42) is reproduced
  and asserted against results/revision/predictions_test.csv, so the test set is identical.
  Candidate variables are chosen by WHEN they are recorded (codebook form/section), never by outcome:
    tier1_antenatal     forms recorded before the delivery admission
    tier2_to_discharge  tier1 + delivery hospitalisation up to discharge
    tier3_sensitivity   tier2 + forms whose window may extend past discharge (hypertension,
                        infections, CMAE complications) -- leakage-prone, sensitivity only
  Every data-driven step (missingness/variance screen, leakage guard, imputation, scaling, RFE,
  L1 path, boosting hyperparameters) is fitted inside training folds; the test set is scored once per model.
Leakage diagnostics: (1) LeakGuard, inside every training fold, drops any variable whose value or
  missingness alone separates the outcome with univariate AUC >= LEAK_AUC (or <= 1 - LEAK_AUC) --
  implausibly strong for a single pre-discharge item, so treated as a probable timing leak;
  (2) leakage_audit.json lists, from the training set only, every variable with univariate
  AUC >= AUDIT_AUC and every form whose wholesale missingness predicts the outcome, for manual timing review.
Outputs: results/expanded/*.json (aggregate only); participant-level predictions are .csv (gitignored).
"""
import json, os, re, sys, time
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.revision_pipeline import SEED, CV, summarise, delong, recalibrate, log, ROOT, OUT as REV_OUT

# DASH files (not distributed): place in data/ or point the environment variables at them
RAW = os.environ.get('NUMOM_RAW', os.path.join(ROOT, 'data', 'nuMoM2b_Dataset_NICHD_Data_Challenge.csv'))
CODEBOOK = os.environ.get('NUMOM_CODEBOOK', os.path.join(ROOT, 'data', 'nuMoM2b_Codebook_NICHD Data Challenge.xlsx'))
CACHE = os.path.join(ROOT, 'data', 'numom2b_expanded.parquet')
OUT = os.environ.get('EXPANDED_OUT', os.path.join(ROOT, 'results', 'expanded'))   # override only for verification reruns
QUICK = bool(os.environ.get('QUICK'))
N_ITER, N_PERM = (3, 5) if QUICK else (15, 100)
LEAK_AUC, AUDIT_AUC = 0.70, 0.65
RFE_K = [10, 25, 50]  # 10 = main paper's RFE size
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- timing classification (codebook forms)
TIER1_FORMS = {'s01', 's02', 'demographics',
               'v1a', 'v1b', 'v1c', 'v1e', 'v1f', 'v1g', 'v1h', 'v1k', 'v1l',          # visit 1 (6-13 wk)
               'v2a', 'v2b', 'v2i', 'v2m',                                              # visit 2 (16-21 wk)
               'v3a', 'v3b', 'v3c', 'v3e', 'v3f', 'v3j', 'v3k', 'v3l',                  # visit 3 (22-29 wk)
               'u02', 'u1c', 'u2a', 'u2b', 'u2c', 'u3a', 'u3b', 'u3c', 'u3d',           # study ultrasounds
               'cla', 'clb', 'cua', 'cub',                                              # prenatal labs, diagnostics, anomaly scans
               'cmb',                                                                   # antepartum hospital encounters
               'placental_analytes', 'physical_activity', 'food_frequency_analysis',
               'sleep_actigraphy', 'sleep_disordered_breathing'}
TIER2_CMA = ('CMAA', 'CMAB', 'CMAC', 'CMAF', 'CMAG', 'CMAH')    # L&D admission, delivery, meds, labs before discharge
TIER2_DERIVED = ('PROM', 'SPON', 'TYPE')                        # pregnancy_outcomes: delivery-time derivations
SENS_FORMS, SENS_CMA = {'cmd', 'cme'}, ('CMAE',)               # windows may run past discharge
# Not used at any tier: CMAD (postpartum complications), CMAI (discharge date), CMAJ (outcome),
# admin forms a*/af*, visit 4 (v4a, postpartum), neonatal follow-up cba/cbb, stillbirth work-up cbc,
# placental pathology cpa (reported after discharge), mixed-timing chart summaries vxx/drugs_in_pregnancy,
# sleep_actigraphy_modified (duplicate), sample_selection/e2c (logistics).
DROP_LABEL = re.compile(r'interval|\bdate\b|\btime\b|hour|minute|specify|serial|name of|order of entry|'
                        r'form version|data entry|validation status|line number|postpartum|post-partum|'
                        r'readmi|discharge|not done \(checkbox\)|not available \(checkbox\)', re.I)
MISSING_CODES = [-7, -8, -9]  # refused / don't know; letter codes (D, R, S, M, E) become NaN on coercion


def inventory():
    cb = pd.read_excel(CODEBOOK, sheet_name=0, header=1)
    cb.columns = ['form', 'var', 'label', 'type', 'unit', 'codelist']
    cb = cb[cb['var'].notna() & (cb['var'] != 'PublicID')].drop_duplicates('var')
    sec = cb['var'].str[:4]
    is_t1 = cb.form.isin(TIER1_FORMS)
    is_t2 = (cb.form.eq('cma') & sec.isin(TIER2_CMA)) | (cb.form.eq('pregnancy_outcomes') & sec.isin(TIER2_DERIVED))
    is_s = cb.form.isin(SENS_FORMS) | (cb.form.eq('cma') & sec.isin(SENS_CMA))
    usable = (~cb.label.astype(str).str.contains(DROP_LABEL)) & (cb.type.eq('Numeric') | cb.codelist.notna())
    usable &= ~cb['var'].str.contains(r'_Complete|_Status|Ver(_\d)?$', regex=True)
    tiers = {'tier1_antenatal': cb['var'][is_t1 & usable].tolist()}
    tiers['tier2_to_discharge'] = tiers['tier1_antenatal'] + cb['var'][is_t2 & usable].tolist()
    tiers['tier3_sensitivity'] = tiers['tier2_to_discharge'] + cb['var'][is_s & usable].tolist()
    labels = dict(zip(cb['var'], cb.label.astype(str)))
    forms = dict(zip(cb['var'], cb.form.astype(str)))
    summary = {'n_codebook_variables': int(len(cb)),
               'excluded_by_label_or_type': int((~usable).sum()),
               'tiers': {k: len(v) for k, v in tiers.items()},
               'per_form': {t: pd.Series(cb.form[cb['var'].isin(v)]).value_counts().to_dict() for t, v in tiers.items()}}
    json.dump({'summary': summary, 'tiers': tiers}, open(os.path.join(OUT, 'variable_inventory.json'), 'w'), indent=1)
    log('inventory', summary['tiers'])
    return tiers, labels, forms


def load(tiers):
    cols = sorted(set(tiers['tier3_sensitivity']))
    if not os.path.exists(CACHE):
        log('reading raw CSV (slow)')
        hdr = pd.read_csv(RAW, nrows=0).columns
        use = ['PublicID', 'CMAJ01'] + [c for c in cols if c in hdr]
        pd.read_csv(RAW, usecols=use, low_memory=False).to_parquet(CACHE)
    df = pd.read_parquet(CACHE)
    assert df.PublicID.is_unique, 'unit of analysis is the participant'
    df = df[df.CMAJ01.notna()].reset_index(drop=True)
    y = (pd.to_numeric(df.CMAJ01, errors='coerce') == 1).astype(int).values
    X = df.drop(columns=['PublicID', 'CMAJ01']).apply(pd.to_numeric, errors='coerce').astype('float32')
    X = X.mask(X.isin(MISSING_CODES))
    # reproduce the main paper's split (same y order, n, seed, stratification) and verify it
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.30, random_state=SEED, stratify=y)  # split-unit:ok one row per participant (asserted above)
    ref = pd.read_csv(os.path.join(REV_OUT, 'predictions_test.csv'))
    assert np.array_equal(ref.y.values, y[idx_te]), 'test split differs from the main analysis'
    ids = df.PublicID.values
    assert not set(ids[idx_tr]) & set(ids[idx_te]), 'participant overlap'
    log(f'train {len(idx_tr)} participants / {y[idx_tr].sum()} events; test {len(idx_te)} / {y[idx_te].sum()}; overlap 0')
    return X, y, idx_tr, idx_te, ref


class Screen(BaseEstimator, TransformerMixin):
    """Fitted on training rows only: drop columns >max_missing missing or near-constant."""
    def __init__(self, max_missing=0.5, min_minority=0.005):
        self.max_missing, self.min_minority = max_missing, min_minority

    def fit(self, X, y=None):
        X = pd.DataFrame(np.asarray(X, dtype='float32'))
        miss = X.isna().mean().values
        top = np.array([X[c].value_counts(normalize=True).iloc[0] if X[c].notna().any() else 1.0 for c in X])
        nonmodal = (1 - top) * (1 - miss)
        self.keep_ = (miss <= self.max_missing) & (nonmodal >= self.min_minority)
        return self

    def transform(self, X):
        return np.asarray(X, dtype='float32')[:, self.keep_]


def univariate_auc(X, y):
    """Column-wise AUC of (median-imputed) values and of missingness indicators; vectorised Mann-Whitney."""
    X = np.asarray(X, dtype='float64')
    miss = np.isnan(X)
    Xf = np.where(miss, np.nanmedian(X, axis=0), X)
    pos, n1, n0 = y == 1, (y == 1).sum(), (y == 0).sum()
    r = rankdata(Xf, axis=0)
    auc_val = (r[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    auc_miss = 0.5 + (miss[pos].mean(0) - miss[~pos].mean(0)) / 2
    return np.nan_to_num(auc_val, nan=0.5), auc_miss


class LeakGuard(BaseEstimator, TransformerMixin):
    """Fitted on training rows only: drop columns whose value or missingness alone predicts the outcome
    with univariate AUC >= auc (or <= 1 - auc)."""
    def __init__(self, auc=LEAK_AUC):
        self.auc = auc

    def fit(self, X, y):
        a_val, a_miss = univariate_auc(X, np.asarray(y))
        self.keep_ = (np.abs(a_val - 0.5) < self.auc - 0.5) & (np.abs(a_miss - 0.5) < self.auc - 0.5)
        return self

    def transform(self, X):
        return np.asarray(X, dtype='float32')[:, self.keep_]


def audit(X, y, cols, labels, forms):
    a_val, a_miss = univariate_auc(X, y)
    flag = [{'var': c, 'form': forms.get(c, ''), 'auc_value': float(v), 'auc_missing': float(m),
             'label': labels.get(c, '')[:110]}
            for c, v, m in zip(cols, a_val, a_miss) if max(abs(v - 0.5), abs(m - 0.5)) >= AUDIT_AUC - 0.5]
    fm = pd.Series(forms).reindex(cols)
    form_missing = {}
    for f in sorted(set(fm.dropna())):
        allmiss = np.isnan(X[:, (fm == f).values]).all(1)
        if 0 < allmiss.mean() < 1:
            form_missing[f] = {'pct_all_missing': float(allmiss.mean()),
                               'readmit_rate_if_missing': float(y[allmiss].mean()),
                               'readmit_rate_if_present': float(y[~allmiss].mean()),
                               'auc_missing': float(0.5 + (allmiss[y == 1].mean() - allmiss[y == 0].mean()) / 2)}
    return {'audit_auc_threshold': AUDIT_AUC, 'leak_guard_threshold': LEAK_AUC,
            'flagged_variables': sorted(flag, key=lambda d: -max(abs(d['auc_value'] - .5), abs(d['auc_missing'] - .5))),
            'form_missingness': form_missing}


def front():
    return [('screen', Screen()), ('guard', LeakGuard())]


def l1_model():
    return Pipeline(front() + [('impute', SimpleImputer(strategy='median', add_indicator=True)),
                     ('scale', StandardScaler()),
                     ('clf', LogisticRegressionCV(Cs=np.logspace(-3, -0.5, 8), penalty='l1', solver='liblinear',
                                                  scoring='roc_auc', cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                                  max_iter=3000, n_jobs=-1))])


def gb_model():
    base = Pipeline(front() + [('clf', HistGradientBoostingClassifier(random_state=SEED, early_stopping=False))])
    space = {'clf__learning_rate': [0.01, 0.03, 0.1], 'clf__max_leaf_nodes': [7, 15, 31],
             'clf__min_samples_leaf': [20, 50, 100, 200], 'clf__l2_regularization': [0.0, 1.0, 10.0],
             'clf__max_iter': [100, 200, 400], 'clf__max_features': [0.1, 0.3, 1.0]}
    return RandomizedSearchCV(base, space, n_iter=N_ITER, scoring='roc_auc', random_state=SEED, n_jobs=-1,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


def _rfe():
    return RFE(LogisticRegression(max_iter=1000, class_weight='balanced'), n_features_to_select=10, step=0.1)


def rfe_lr_model():
    base = Pipeline(front() + [('impute', SimpleImputer(strategy='median', add_indicator=True)),
                               ('scale', StandardScaler()), ('rfe', _rfe()),
                               ('clf', LogisticRegression(max_iter=2000, random_state=SEED))])
    space = {'rfe__n_features_to_select': RFE_K, 'clf__C': list(np.logspace(-3, 1, 6)),
             'clf__class_weight': ['balanced', None]}
    return RandomizedSearchCV(base, space, n_iter=min(N_ITER, 36), scoring='roc_auc', random_state=SEED, n_jobs=-1,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


def rfe_gb_model():
    base = Pipeline(front() + [('impute', SimpleImputer(strategy='median', add_indicator=True)),
                               ('scale', StandardScaler()), ('rfe', _rfe()),
                               ('clf', HistGradientBoostingClassifier(random_state=SEED, early_stopping=False))])
    space = {'rfe__n_features_to_select': RFE_K, 'clf__learning_rate': [0.01, 0.03, 0.1],
             'clf__max_leaf_nodes': [3, 7, 15], 'clf__min_samples_leaf': [20, 50, 100],
             'clf__l2_regularization': [0.0, 1.0, 10.0], 'clf__max_iter': [100, 200, 400]}
    return RandomizedSearchCV(base, space, n_iter=N_ITER, scoring='roc_auc', random_state=SEED, n_jobs=-1,
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED))


def fitted_params(m):
    if isinstance(m, RandomizedSearchCV):
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in m.best_params_.items()}
    return {'C': float(m.named_steps['clf'].C_[0]), 'n_nonzero': int((m.named_steps['clf'].coef_ != 0).sum())}


def fixed(m):
    """Same model with the hyperparameters selected on the full training set (for permutation control)."""
    if isinstance(m, RandomizedSearchCV):
        return clone(m.best_estimator_)
    p = clone(l1_model())
    p.steps[-1] = ('clf', LogisticRegression(C=m.named_steps['clf'].C_[0], penalty='l1', solver='liblinear', max_iter=3000))
    return p


def stability(Xtr, ytr, model, names, n=100):
    """L1 selection frequency over half-subsamples of the training set (screen refitted each time)."""
    rng, counts = np.random.RandomState(SEED), pd.Series(0.0, index=names)
    for _ in range(10 if QUICK else n):
        i = np.concatenate([rng.choice(np.where(ytr == k)[0], (ytr == k).sum() // 2, replace=False) for k in (0, 1)])
        m = fixed(model).fit(Xtr[i], ytr[i])
        kept = np.array(names)[m.named_steps['screen'].keep_][m.named_steps['guard'].keep_]
        coef = m.named_steps['clf'].coef_[0]
        nz = set(kept[coef[:len(kept)] != 0])                           # value columns
        ind = m.named_steps['impute'].indicator_
        if ind is not None:                                             # missing-indicator columns -> parent variable
            nz |= set(kept[ind.features_][coef[len(kept):] != 0])
        counts[list(nz)] += 1
    return (counts / (10 if QUICK else n)).sort_values(ascending=False)


def main():
    t0 = time.time()
    tiers, labels, forms = inventory()
    X, y, itr, ite, ref = load(tiers)
    ytr, yte = y[itr], y[ite]
    res = {'_note': 'EXPLORATORY, POST HOC analysis (added after the main revision results were known). '
                    'An earlier full run (L1_LR, HGB only) was stopped after scoring tier 1 on the test set; the RFE models and the '
                    'univariate leakage guard were then added at the author\'s request, with those tier-1 test results visible. '
                    'A smoke test with reduced budgets (QUICK) also scored the test set first; no design change followed it. '
                    'Source: src/expanded_features.py', 'tiers': {}}
    import subprocess
    git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain', 'src/expanded_features.py'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    res.update({'script': 'src/expanded_features.py', 'git_hash': git + ('+uncommitted' if dirty else ''),
                'unit_of_analysis': 'participant', 'splitter': 'train_test_split (test_size 0.30, stratified, seed as main analysis)',
                'groups_passed': False, 'groups_note': 'one row per participant (asserted in load())',
                'n_train_units': int(len(itr)), 'n_test_units': int(len(ite)),
                'n_overlap_units': int(len(set(itr) & set(ite)))})
    test_pred, oof_pred = {'y': yte}, {'y': ytr}
    for tier, cols in tiers.items():
        cols = [c for c in cols if c in X.columns]
        Xtr, Xte = X.iloc[itr][cols].values, X.iloc[ite][cols].values
        res['tiers'][tier] = {'n_candidate_variables': len(cols)}
        aud = audit(Xtr, ytr, cols, labels, forms)
        json.dump(aud, open(os.path.join(OUT, f'leakage_audit_{tier}.json'), 'w'), indent=1)
        res['tiers'][tier]['leakage_audit'] = {'n_flagged_variables': len(aud['flagged_variables']),
                                               'file': f'results/expanded/leakage_audit_{tier}.json'}
        log(f'{tier}: audit flagged {len(aud["flagged_variables"])} variables')
        for name, make in (('L1_LR', l1_model), ('HGB', gb_model), ('RFE_LR', rfe_lr_model), ('RFE_HGB', rfe_gb_model)):
            key = f'{tier}__{name}'
            log(f'{key}: {len(cols)} candidates; nested CV')
            p_oof = cross_val_predict(make(), Xtr, ytr, cv=CV, method='predict_proba', n_jobs=1)[:, 1]
            m = make().fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            best = m.best_estimator_ if isinstance(m, RandomizedSearchCV) else m
            screen, guard = best.named_steps['screen'], best.named_steps['guard']
            kept = np.array(cols)[screen.keep_]
            d, z, pv = delong(yte, p, ref['LR_tuned'].values)
            r = {'n_after_screen_full_train': int(screen.keep_.sum()),
                 'leak_guard_dropped_full_train': kept[~guard.keep_].tolist(), 'params': fitted_params(m),
                 'oof_roc_auc': float(roc_auc_score(ytr, p_oof)),
                 'test': summarise(yte, p, {'default_0.5': 0.5}),
                 'delong_vs_LR_tuned_18var': {'diff': d, 'z': z, 'p': pv}}
            perm = []
            rng = np.random.RandomState(SEED)
            for _ in range(N_PERM):  # label-permutation control with the selected hyperparameters
                perm.append(roc_auc_score(yte, fixed(m).fit(Xtr, rng.permutation(ytr)).predict_proba(Xte)[:, 1]))
            r['permutation_control'] = {'n': N_PERM, 'perm_auc_range': [float(min(perm)), float(max(perm))],
                                        'p': float((1 + sum(a >= r['test']['roc_auc'] for a in perm)) / (N_PERM + 1))}
            if 'rfe' in best.named_steps:  # variables RFE kept on the full training set (indicator columns -> parent)
                kept2 = kept[guard.keep_]
                ind = best.named_steps['impute'].indicator_
                names = list(kept2) + ([f'{v}__missing' for v in kept2[ind.features_]] if ind is not None else [])
                r['rfe_selected'] = [{'var': v, 'label': labels.get(v.split('__')[0], '')[:110]}
                                     for v, k in zip(names, best.named_steps['rfe'].support_) if k]
            if name == 'L1_LR':
                st = stability(Xtr, ytr, m, cols)
                r['stability_top30'] = [{'var': v, 'freq': float(f), 'label': labels.get(v, '')[:110]}
                                        for v, f in st.head(30).items()]
            res['tiers'][tier][name] = r
            test_pred[key], oof_pred[key] = p, p_oof
            log(f'  {key}: OOF {r["oof_roc_auc"]:.3f}, test {r["test"]["roc_auc"]:.3f}, '
                f'perm p {r["permutation_control"]["p"]:.3f}, {time.time() - t0:.0f}s')
            json.dump(res, open(os.path.join(OUT, 'expanded.json'), 'w'), indent=1, default=float)
            pd.DataFrame(test_pred).to_csv(os.path.join(OUT, 'predictions_test.csv'), index=False)
            pd.DataFrame(oof_pred).to_csv(os.path.join(OUT, 'predictions_train_oof.csv'), index=False)
    log(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
