"""
Revision analysis for MLHEALTH-100198 (single pinned script; every reported number comes from here).

Unit of analysis: participant (one row per PublicID, verified below). Split: stratified 70/30, seed 42.
Every fitted step (imputation, scaling, RFE, SMOTE, hyperparameters, thresholds, recalibration) is
learned inside the training set. Thresholds and recalibration maps are fit on 5-fold out-of-fold (OOF)
training predictions, then applied unchanged to the test set.

Stages (each writes results/revision/<stage>.json; rerun skips finished stages unless --force):
  models     original-configuration and tuned ML, FLAML (default + imbalance-aware), ensembles
  autogluon  AutoGluon best_quality (separate process)
  evaluate   test metrics with bootstrap CIs, operating points, recalibration, DCA, DeLong, controls
  datalimit  learning curve + extended/discharge feature sets

Usage: .venv/bin/python src/revision_pipeline.py [stage ...] [--force]
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (GradientBoostingClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_predict, learning_curve, train_test_split)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
from predictors import predictors as CANDIDATES  # 18 sociodemographic candidates
# Data-limitation feature sets (labels verified against the nuMoM2b codebook)
ENROLLMENT_EXTRA = ['Age_at_V1', 'AgeCat_V1',
                    'V1AF04',                     # marital status
                    'V1AD15',                     # asthma
                    'V1AE01',                     # number of pregnancies
                    'V1AG04', 'V1AG05',           # ever used tobacco; smoked in 3 months before pregnancy
                    'V1AD02a', 'V1AD02b', 'V1AD02c', 'V1AD02d', 'V1AD02e']  # preconception counselling topics
DISCHARGE_EXTRA = ['CMAC15',                      # cesarean delivery
                   'CMAC12',                      # operative vaginal delivery attempted
                   'CMAE03',                      # diabetes ever diagnosed
                   'CMAE04a1a', 'CMAE04a1b',      # treated for depression before / during pregnancy
                   'CMAE04a2a', 'CMAE04a2b',      # treated for anxiety before / during pregnancy
                   'A09A03a',                     # preterm birth
                   'SGA_alex']                    # small for gestational age
# No time window in the codebook; abstracted on the same form as the outcome, so may include
# postpartum onset -> excluded from the primary discharge set, included only as a sensitivity analysis.
HYPERTENSION_UNTIMED = ['CMAE01', 'A09A03b']

DATA = os.path.join(ROOT, 'data', 'numom2b_analysis_subset.csv')
QUICK = bool(os.environ.get('QUICK'))  # smoke test: tiny budgets, scratch output
OUT = os.environ.get('REVISION_OUT') or os.path.join(ROOT, 'results', 'revision')
SEED = 42
SPLIT = {}  # filled by the split; recorded in every result file's _provenance
N_BOOT = 50 if QUICK else 1000
CV = StratifiedKFold(5, shuffle=True, random_state=SEED)
FLAML_DEFAULT_S, FLAML_TUNED_S, AUTOGLUON_S, AUTOGLUON_OOF_S = (15, 15, 240, 120) if QUICK else (300, 900, 900, 300)
os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---------------------------------------------------------------- data
def load(features=CANDIDATES):
    df = pd.read_csv(DATA, low_memory=False)
    assert df.PublicID.is_unique, 'unit of analysis is the participant; rows must be unique per PublicID'
    df = df[df.CMAJ01.notna()].copy()
    # structural (skip-pattern) missingness is 'not applicable', not unknown: recode before imputation
    for c in ('SmokeCat2', 'SmokeCat3', 'V1AG04', 'V1AG05'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df.loc[df.SmokeCat3.isna() & (df.SmokeCat2 == 2), 'SmokeCat3'] = 0  # no smoking in prior 3 months
    df.loc[df.V1AG05.isna() & (df.V1AG04 == 2), 'V1AG05'] = 2           # never used tobacco -> did not smoke
    y = (df.CMAJ01 == 1).astype(int).values
    X = df[features].apply(pd.to_numeric, errors='coerce')
    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(X, y, df.PublicID.values, test_size=0.30,
                                                      random_state=SEED, stratify=y)
    assert not set(idtr) & set(idte), 'participant overlap between train and test'
    SPLIT.update(splitter='train_test_split(test_size=0.30, stratify=y, random_state=SEED)', groups_passed=False,
                 unit_of_analysis='participant (one row each)', n_train_units=len(set(idtr)), n_test_units=len(set(idte)),
                 n_overlap_units=len(set(idtr) & set(idte)))
    log(f'train {len(ytr)} rows / {len(set(idtr))} participants / {ytr.sum()} events; '
        f'test {len(yte)} rows / {len(set(idte))} participants / {yte.sum()} events; overlap 0')
    return Xtr, Xte, ytr, yte


NOMINAL = ['CRace', 'Race', 'eRace', 'eHispanic', 'V1AF04']  # unordered codes: one-hot, never numeric


def prep(indicator=False):
    """Fitted inside each training fold: nominal -> mode impute + one-hot (categories <20 pooled);
    everything else -> median impute (+ missing indicators) + standardise."""
    return ColumnTransformer([
        ('nom', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                          ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20,
                                                   sparse_output=False))]),
         lambda X: [c for c in X.columns if c in NOMINAL]),
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=indicator)),
                          ('scale', StandardScaler())]),
         lambda X: [c for c in X.columns if c not in NOMINAL])])


def pipe(clf, smote=True, rfe=True, indicator=False):
    steps = [('prep', prep(indicator))]
    if rfe:
        steps.append(('rfe', RFE(LogisticRegression(max_iter=1000, class_weight='balanced'),
                                 n_features_to_select=10)))
    steps.append(('sampler', SMOTE(random_state=SEED) if smote else 'passthrough'))
    steps.append(('clf', clf))
    return Pipeline(steps)


SAMPLERS = [SMOTE(random_state=SEED), 'passthrough']
SEARCH = {
    'LR': (LogisticRegression(max_iter=2000, random_state=SEED),
           {'clf__C': np.logspace(-3, 2, 12), 'clf__class_weight': ['balanced', None],
            'sampler': SAMPLERS}, 40),
    'RF': (RandomForestClassifier(random_state=SEED, n_jobs=-1),
           {'clf__n_estimators': [200, 500], 'clf__max_depth': [3, 5, 8, None],
            'clf__min_samples_leaf': [1, 5, 20, 50], 'clf__max_features': ['sqrt', 0.5],
            'clf__class_weight': ['balanced', 'balanced_subsample', None], 'sampler': SAMPLERS}, 40),
    'GB': (GradientBoostingClassifier(random_state=SEED),
           {'clf__n_estimators': [50, 100, 300], 'clf__learning_rate': [0.01, 0.05, 0.1],
            'clf__max_depth': [2, 3], 'clf__subsample': [0.7, 1.0],
            'clf__min_samples_leaf': [1, 20, 50], 'sampler': SAMPLERS}, 40),
}


def tune(name, Xtr, ytr, rfe=True, indicator=False):
    est, grid, n_iter = SEARCH[name]
    rs = RandomizedSearchCV(pipe(est, rfe=rfe, indicator=indicator), grid, n_iter=3 if QUICK else n_iter,
                            scoring='roc_auc', cv=CV, random_state=SEED, n_jobs=-1, refit=True)
    rs.fit(Xtr, ytr)
    params = {k: (type(v).__name__ if not isinstance(v, (int, float, str, type(None))) else v)
              for k, v in rs.best_params_.items()}
    log(f'  tuned {name}: CV AUC {rs.best_score_:.3f} {params}')
    return rs.best_estimator_, {'cv_auc': rs.best_score_, 'best_params': params}


def oof(model, Xtr, ytr):
    return cross_val_predict(clone(model), Xtr, ytr, cv=CV, method='predict_proba', n_jobs=-1)[:, 1]


# ---------------------------------------------------------------- stage: models
def stage_models():
    Xtr, Xte, ytr, yte = load()
    test, train_oof, info = {'y': yte}, {'y': ytr}, {}

    def record(name, model, extra=None, p_oof=None):
        t0 = time.time()
        if p_oof is None:
            p_oof = oof(model, Xtr, ytr)
        model.fit(Xtr, ytr)
        info[name] = {'fit_seconds': time.time() - t0, **(extra or {})}
        if 'rfe' in model.named_steps:
            names = model.named_steps['prep'].get_feature_names_out()
            info[name]['rfe_selected'] = list(names[model.named_steps['rfe'].support_])
        test[name], train_oof[name] = model.predict_proba(Xte)[:, 1], p_oof
        log(f'  {name}: test AUC {roc_auc_score(yte, test[name]):.3f}')
        return model

    log('original configuration (as submitted)')
    orig = {'LR_orig': LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=SEED),
            'RF_orig': RandomForestClassifier(100, class_weight='balanced', random_state=SEED, n_jobs=-1),
            'GB_orig': GradientBoostingClassifier(random_state=SEED)}
    fitted = {k: record(k, pipe(v)) for k, v in orig.items()}
    lr_nosmote = record('LR_orig_noSMOTE', pipe(clone(orig['LR_orig']), smote=False))

    log('tuned ML (RandomizedSearchCV, 5-fold, ROC-AUC; sampler and class weight tunable)')
    tuned = {}
    for n in SEARCH:
        best, meta = tune(n, Xtr, ytr)
        tuned[n + '_tuned'] = record(n + '_tuned', best, meta)

    log('ensembles (original-configuration and tuned bases)')
    for tag, bases in (('orig', fitted), ('tuned', tuned)):
        est = [(k, clone(v)) for k, v in bases.items()]
        record(f'Stacking_{tag}', Pipeline([('m', StackingClassifier(
            est, final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
            cv=CV, stack_method='predict_proba', n_jobs=-1))]),
            {'meta_learner': 'LogisticRegression(C=1.0, class_weight=balanced)', 'cv': 'stratified 5-fold',
             'base_models': [k for k, _ in est]})
        record(f'SoftVote_{tag}', Pipeline([('m', VotingClassifier(
            [(k, CalibratedClassifierCV(v, method='isotonic', cv=CV)) for k, v in est], voting='soft'))]),
            {'calibration': 'isotonic, CalibratedClassifierCV stratified 5-fold per base model',
             'base_models': [k for k, _ in est]})

    # AutoML gets the same imputation + scaling, fitted on training only (no RFE/SMOTE unless stated)
    pre = prep().fit(Xtr)
    Atr, Ate = pre.transform(Xtr), pre.transform(Xte)

    from flaml import AutoML
    def flaml_run(name, budget, X_fit, y_fit, **kw):
        aml = AutoML()
        t0 = time.time()
        aml.fit(X_fit, y_fit, task='classification', metric='roc_auc', time_budget=budget,
                seed=SEED, verbose=0, **kw)
        secs = time.time() - t0
        base = aml.model.estimator
        fit_kw = {}
        if 'sample_weight' in kw:
            p_oof = np.zeros(len(ytr))
            for a, b in CV.split(Atr, ytr):  # split-unit:ok inner CV within training set; one row per participant (asserted in load)
                m = clone(base).fit(Atr[a], ytr[a], sample_weight=kw['sample_weight'][a])
                p_oof[b] = m.predict_proba(Atr[b])[:, 1]
        else:
            # OOF on the original (non-SMOTE) training rows; SMOTE re-applied inside each fold if used
            p_oof = cross_val_predict(Pipeline([('s', SMOTE(random_state=SEED) if len(y_fit) != len(ytr)
                                                 else 'passthrough'), ('m', clone(base))]),
                                      Atr, ytr, cv=CV, method='predict_proba')[:, 1]
        test[name], train_oof[name] = aml.predict_proba(Ate)[:, 1], p_oof
        info[name] = {'fit_seconds': secs, 'budget_s': budget, 'best_estimator': aml.best_estimator,
                      'best_config': aml.best_config, 'best_cv_loss': aml.best_loss,
                      'eval_method': kw.get('eval_method', 'auto'),
                      'sample_weight': 'balanced' if 'sample_weight' in kw else None,
                      'smote': len(y_fit) != len(ytr)}
        log(f'  {name}: {aml.best_estimator}, test AUC {roc_auc_score(yte, test[name]):.3f}')

    log('FLAML default (as submitted: SMOTE-resampled training data, holdout, 300 s)')
    Xs, ys = SMOTE(random_state=SEED).fit_resample(Atr, ytr)
    flaml_run('FLAML_default', FLAML_DEFAULT_S, Xs, ys)
    log('FLAML imbalance-aware (5-fold CV, balanced sample weights, 900 s)')
    w = np.where(ytr == 1, (ytr == 0).sum() / ytr.sum(), 1.0)
    flaml_run('FLAML_tuned', FLAML_TUNED_S, Atr, ytr, eval_method='cv', n_splits=5, sample_weight=w)

    # participant-level predictions are derived data: kept locally (*.csv is gitignored)
    pd.DataFrame(test).to_csv(os.path.join(OUT, 'predictions_test.csv'), index=False)
    pd.DataFrame(train_oof).to_csv(os.path.join(OUT, 'predictions_train_oof.csv'), index=False)
    return {'models': info, 'n_train': int(len(ytr)), 'n_test': int(len(yte)),
            'events_train': int(ytr.sum()), 'events_test': int(yte.sum())}


# ---------------------------------------------------------------- stage: autogluon
def stage_autogluon():
    """Own stage (fresh process): torch/CatBoost OpenMP deadlocks once sklearn/FLAML pools exist."""
    from autogluon.tabular import TabularPredictor
    Xtr, Xte, ytr, yte = load()
    pre = prep().fit(Xtr)
    Atr, Ate = pre.transform(Xtr), pre.transform(Xte)
    run_dir = os.path.join(ROOT, 'data', f"autogluon_rev_{time.strftime('%Y%m%d_%H%M%S')}")

    def frame(A):  # fold-fitted encoders may yield different column counts (rare-category pooling)
        return pd.DataFrame(A, columns=[f'f{i}' for i in range(A.shape[1])])

    def fit_ag(A, y, budget, tag):
        return TabularPredictor('y', eval_metric='roc_auc', path=os.path.join(run_dir, tag), verbosity=1).fit(
            frame(A).assign(y=y), presets='best_quality', time_limit=budget)

    log(f'AutoGluon (best_quality, ROC-AUC, {AUTOGLUON_S} s)')
    t0 = time.time()
    ag = fit_ag(Atr, ytr, AUTOGLUON_S, 'full')
    secs = time.time() - t0
    p_test = ag.predict_proba(frame(Ate))[1].values
    lb = ag.leaderboard(silent=True)
    log(f'  AutoGluon: {ag.model_best}, test AUC {roc_auc_score(yte, p_test):.3f}')
    # AutoGluon's own OOF is optimistic (best of ~100 models + ensemble weights fit on the same OOF),
    # so thresholds/recalibration use honest OOF: AutoGluon refit inside each outer training fold.
    log(f'AutoGluon honest OOF: 5 outer folds x {AUTOGLUON_OOF_S} s')
    p_oof = np.zeros(len(ytr))
    for k, (a, b) in enumerate(CV.split(Xtr, ytr)):  # split-unit:ok inner CV within training set; one row per participant (asserted in load)
        pk = prep().fit(Xtr.iloc[a])
        m = fit_ag(pk.transform(Xtr.iloc[a]), ytr[a], AUTOGLUON_OOF_S, f'fold{k}')
        p_oof[b] = m.predict_proba(frame(pk.transform(Xtr.iloc[b])))[1].values
    pd.DataFrame({'AutoGluon': p_test}).to_csv(os.path.join(OUT, 'predictions_test_autogluon.csv'), index=False)
    pd.DataFrame({'AutoGluon': p_oof}).to_csv(os.path.join(OUT, 'predictions_train_oof_autogluon.csv'), index=False)
    self_oof = roc_auc_score(ytr, ag.predict_proba_oof()[1].values)
    log(f'  AutoGluon OOF AUC: internal {self_oof:.3f}, honest outer-fold {roc_auc_score(ytr, p_oof):.3f}')
    return {'fit_seconds': secs, 'budget_s': AUTOGLUON_S, 'preset': 'best_quality',
            'best_model': ag.model_best, 'n_models': int(len(lb)), 'models_trained': lb.model.tolist(),
            'internal_oof_auc': self_oof, 'internal_l1_score_val_median': float(lb[lb.stack_level == 1].score_val.median()),
            'oof': f'honest: AutoGluon refit in each of 5 outer training folds ({AUTOGLUON_OOF_S} s each)'}


# ---------------------------------------------------------------- metrics
def cal_slope_intercept(y, p):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    import statsmodels.api as sm
    if np.ptp(lp) < 1e-9:
        return float('nan'), float('nan')  # constant predictions: slope undefined
    slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0).params[1]
    intercept = sm.GLM(y, np.ones_like(lp), offset=lp, family=sm.families.Binomial()).fit().params[0]
    if not np.isfinite(intercept) or abs(intercept) > 20:  # GLM failed to converge (degenerate predictions)
        intercept = float('nan')
    return slope, intercept


def at_threshold(y, p, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {'threshold': float(t), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'sensitivity': tp / max(tp + fn, 1), 'specificity': tn / max(tn + fp, 1),
            'ppv': tp / (tp + fp) if tp + fp else None, 'npv': tn / (tn + fn) if tn + fn else None,
            'pct_flagged': (tp + fp) / len(y)}


def boot_ci(y, p, fns, n=N_BOOT):
    """Stratified bootstrap (events and non-events resampled separately), percentile 95% CI."""
    rng = np.random.RandomState(SEED)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    draws = {k: [] for k in fns}
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, f in fns.items():
            draws[k].append(f(y[i], p[i]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def summarise(y, p, thresholds):
    fns = {'roc_auc': roc_auc_score, 'pr_auc': average_precision_score, 'brier': brier_score_loss}
    for name, t in thresholds.items():
        fns[f'sens@{name}'] = lambda a, b, t=t: ((b >= t) & (a == 1)).sum() / (a == 1).sum()
        fns[f'spec@{name}'] = lambda a, b, t=t: ((b < t) & (a == 0)).sum() / (a == 0).sum()
    slope, intercept = cal_slope_intercept(y, p)
    return {'roc_auc': roc_auc_score(y, p), 'pr_auc': average_precision_score(y, p),
            'brier': brier_score_loss(y, p), 'brier_null': brier_score_loss(y, np.full_like(p, y.mean())),
            'cal_slope': slope, 'cal_intercept': intercept, 'mean_pred': float(p.mean()),
            'ci95': boot_ci(y, p, fns),
            'operating_points': {k: at_threshold(y, p, t) for k, t in thresholds.items()}}


def oof_thresholds(y_oof, p_oof, flag_rate):
    """Thresholds chosen on training OOF predictions only (never on the test set)."""
    out = {'default_0.5': 0.5}
    for s in (0.8, 0.7):
        # highest threshold whose OOF sensitivity is >= s
        ts = np.unique(p_oof)[::-1]
        sens = np.array([((p_oof >= t) & (y_oof == 1)).sum() / y_oof.sum() for t in ts])
        out[f'oof_sens{int(s * 100)}'] = float(ts[np.argmax(sens >= s)])
    out[f'oof_flag{int(flag_rate * 100)}pct'] = float(np.quantile(p_oof, 1 - flag_rate))
    return out


def recalibrate(y_oof, p_oof, p_test):
    lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    platt = LogisticRegression(C=1e6).fit(lo(p_oof)[:, None], y_oof)
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1).fit(p_oof, y_oof)
    return {'platt': platt.predict_proba(lo(p_test)[:, None])[:, 1], 'isotonic': iso.predict(p_test)}


def net_benefit(y, p, ts):
    n = len(y)
    return [float(((p >= t) & (y == 1)).sum() / n - ((p >= t) & (y == 0)).sum() / n * t / (1 - t)) for t in ts]


# DeLong (Sun & Xu 2014), copied from src/delong_auc_comparison.py
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    T2 = np.empty(N); T2[J] = T; return T2


def delong(y, p1, p2):
    order = np.argsort(-y); m = int(y.sum()); P = np.vstack((p1, p2))[:, order]; n = P.shape[1] - m
    tx = np.array([_midrank(r[:m]) for r in P]); ty = np.array([_midrank(r[m:]) for r in P])
    tz = np.array([_midrank(r) for r in P])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    cov = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1.0 - (tz[:, m:] - ty) / m) / n
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = (aucs[0] - aucs[1]) / se
    return float(aucs[0] - aucs[1]), float(z), float(2 * stats.norm.sf(abs(z)))


def holm(pvals):
    keys = sorted(pvals, key=pvals.get); m = len(keys); out, run = {}, 0
    for i, k in enumerate(keys):
        run = max(run, min(1, (m - i) * pvals[k])); out[k] = run
    return out


# ---------------------------------------------------------------- stage: evaluate
def stage_evaluate():
    te = pd.read_csv(os.path.join(OUT, 'predictions_test.csv'))
    tr = pd.read_csv(os.path.join(OUT, 'predictions_train_oof.csv'))
    if os.path.exists(os.path.join(OUT, 'predictions_test_autogluon.csv')):
        te = te.join(pd.read_csv(os.path.join(OUT, 'predictions_test_autogluon.csv')))
        tr = tr.join(pd.read_csv(os.path.join(OUT, 'predictions_train_oof_autogluon.csv')))
    y, yo = te.pop('y').values, tr.pop('y').values
    # matched flag rate: the share the submitted LR flagged at 0.5, re-derived from its OOF scores
    flag_rate = float((tr['LR_orig'] >= 0.5).mean())
    # Added after observing that OOF-quantile thresholds did not equalise test workload: flag the same top share
    # of the test population for every model (label-free: uses only the score distribution).
    test_flag = float((te['LR_orig'] >= 0.5).mean())
    res_flag = {'test_flag_rate_reference': test_flag}
    res = {'flag_rate_reference': flag_rate, **res_flag, 'models': {}}
    ts_dca = np.round(np.arange(0.005, 0.105, 0.005), 3)
    res['dca_thresholds'] = ts_dca.tolist()
    res['dca_treat_all'] = [float(y.mean() - (1 - y.mean()) * t / (1 - t)) for t in ts_dca]
    for m in te.columns:
        p, po = te[m].values, tr[m].values
        ths = oof_thresholds(yo, po, flag_rate)
        ths[f'testtop_flag{int(round(test_flag * 100))}pct'] = float(np.quantile(p, 1 - test_flag))
        r = summarise(y, p, ths)
        r['oof_auc'] = roc_auc_score(yo, po)
        r['recalibrated'] = {}
        for k, pc in recalibrate(yo, po, p).items():
            s, i = cal_slope_intercept(y, pc)
            r['recalibrated'][k] = {'brier': brier_score_loss(y, pc), 'cal_slope': s, 'cal_intercept': i,
                                    'mean_pred': float(pc.mean()), 'roc_auc': roc_auc_score(y, pc),
                                    'net_benefit': net_benefit(y, pc, ts_dca)}
        res['models'][m] = r
        log(f'{m:18s} AUC {r["roc_auc"]:.3f} {r["ci95"]["roc_auc"]} Brier {r["brier"]:.4f} '
            f'sens@0.5 {r["operating_points"]["default_0.5"]["sensitivity"]:.3f}')
    ref = [c for c in te.columns if c != 'LR_tuned']
    raw = {f'LR_tuned_vs_{c}': delong(y, te['LR_tuned'].values, te[c].values) for c in ref}
    adj = holm({k: v[2] for k, v in raw.items()})
    res['delong_vs_LR_tuned'] = {k: {'delta_auc': v[0], 'z': v[1], 'p': v[2], 'p_holm': adj[k]}
                                 for k, v in raw.items()}
    # control: tuned-LR configuration retrained on permuted training labels (chance reference)
    Xtr, Xte, ytr, yte = load()
    assert (yte == y).all(), 'prediction file does not match the current split'
    rng = np.random.RandomState(SEED)
    base = pipe(LogisticRegression(max_iter=2000), smote=False)
    perm = [roc_auc_score(yte, clone(base).fit(Xtr, rng.permutation(ytr)).predict_proba(Xte)[:, 1])
            for _ in range(200)]
    res['permutation_control'] = {'n': 200, 'auc_mean': float(np.mean(perm)),
                                  'auc_95pct_range': [float(np.percentile(perm, 2.5)),
                                                      float(np.percentile(perm, 97.5))],
                                  'p_LR_tuned': float((np.sum(np.array(perm) >= res['models']['LR_tuned']['roc_auc']) + 1) / 201)}
    log('permutation control', res['permutation_control'])
    return res


# ---------------------------------------------------------------- stage: data limitation
def stage_datalimit():
    out = {}
    lc_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    Xtr, _, ytr, _ = load()
    lr = pipe(LogisticRegression(max_iter=2000, class_weight='balanced'), smote=False)
    sizes, _, scores = learning_curve(lr, Xtr, ytr, train_sizes=np.linspace(0.2, 1.0, 5), cv=lc_cv,
                                      scoring='roc_auc', n_jobs=-1, random_state=SEED, shuffle=True)
    out['learning_curve_LR'] = {'train_sizes': sizes.tolist(), 'auc_mean': scores.mean(1).tolist(),
                                'auc_sd': scores.std(1).tolist(), 'cv': '5x5 repeated stratified, training set only'}
    log('learning curve', out['learning_curve_LR']['auc_mean'])
    sets = {'sociodemographic': CANDIDATES,
            'enrollment_extended': CANDIDATES + ENROLLMENT_EXTRA,
            'discharge_available': CANDIDATES + ENROLLMENT_EXTRA + DISCHARGE_EXTRA,
            'sensitivity_discharge_plus_untimed_hypertension':
                CANDIDATES + ENROLLMENT_EXTRA + DISCHARGE_EXTRA + HYPERTENSION_UNTIMED}
    for name, feats in sets.items():
        Xtr, Xte, ytr, yte = load(feats)
        out[name] = {'features': feats, 'missing_pct': (Xtr.isna().mean() * 100).round(1).to_dict()}
        for m in ('LR', 'GB'):
            best, meta = tune(m, Xtr, ytr, rfe=False, indicator=True)
            if name == 'sociodemographic' and m == 'LR':
                # post hoc (added after the pre-specified RFE curve was seen): tuned configuration, no RFE
                s2, _, sc2 = learning_curve(clone(best), Xtr, ytr, train_sizes=np.linspace(0.2, 1.0, 5), cv=lc_cv,
                                            scoring='roc_auc', n_jobs=-1, random_state=SEED, shuffle=True)
                out['learning_curve_LR_tuned_noRFE_posthoc'] = {'train_sizes': s2.tolist(), 'auc_mean': sc2.mean(1).tolist(),
                                                               'auc_sd': sc2.std(1).tolist(), 'config': meta['best_params']}
            p = best.predict_proba(Xte)[:, 1]
            out[name][m] = {**meta, 'test_auc': roc_auc_score(yte, p), 'test_pr_auc': average_precision_score(yte, p),
                            'test_brier': brier_score_loss(yte, p),
                            'ci95': boot_ci(yte, p, {'roc_auc': roc_auc_score, 'pr_auc': average_precision_score})}
            log(f'  {name} {m}: test AUC {out[name][m]["test_auc"]:.3f} {out[name][m]["ci95"]["roc_auc"]}')
    return out


def main():
    stages = {'models': stage_models, 'autogluon': stage_autogluon, 'evaluate': stage_evaluate,
              'datalimit': stage_datalimit}
    force = '--force' in sys.argv
    todo = [a for a in sys.argv[1:] if a in stages] or list(stages)
    if len(todo) > 1:  # one fresh process per stage (avoids the OpenMP deadlock above)
        import subprocess
        for s in todo:
            subprocess.run([sys.executable, __file__, s] + (['--force'] if force else []), check=True)
        return
    for s in todo:
        path = os.path.join(OUT, f'{s}.json')
        if os.path.exists(path) and not force:
            log(f'skip {s} (exists)'); continue
        log(f'=== stage {s}')
        res = stages[s]()
        import subprocess
        git = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        if subprocess.run(['git', 'status', '--porcelain', 'src/revision_pipeline.py'], capture_output=True, text=True, cwd=ROOT).stdout.strip():
            git += '+uncommitted'
        res['_provenance'] = {'script': 'src/revision_pipeline.py', 'git_hash': git, 'seed': SEED,
                              'finished': time.strftime('%Y-%m-%d %H:%M:%S'), **SPLIT}
        json.dump(res, open(path, 'w'), indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
        log(f'wrote {path}')


if __name__ == '__main__':
    main()
