"""EXPLORATORY, POST HOC: AutoGluon on the tier-2 (up to discharge) variables chosen by L1 stability selection.

Decided after the expanded-predictor results (src/expanded_features.py) were known, at the author's request,
so this is test-set-informed: the tier, the selector and the 0.5 threshold were chosen with those results visible.
To keep the selection honest, stability selection is REFITTED inside every training set it is used for:
each of the 5 outer folds (honest OOF predictions, used for Platt recalibration and the economic analysis)
and the full training set (the one model scored on the test set). Unit of analysis = participant; split asserted
by expanded_features.load().

Two stages in separate processes (sklearn/joblib pools then AutoGluon can deadlock, see revision_pipeline):
  python3.12 src/expanded_autogluon.py select   -> results/expanded/autogluon_selection.json
  python3.12 src/expanded_autogluon.py fit      -> results/expanded/autogluon.json (+ gitignored prediction CSVs)
"""
import json, os, sys, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess
from src.revision_pipeline import SEED, CV, summarise, delong, log, ROOT
from src.expanded_features import inventory, load, l1_model, fixed, OUT

TIER, THRESH, N_SUB, AG_S = 'tier2_to_discharge', 0.5, 100, 600
GIT = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
SEL = os.path.join(OUT, 'autogluon_selection.json')


def _nonzero(model, Xs, ys, names):
    m = model.fit(Xs, ys)
    kept = np.array(names)[m.named_steps['screen'].keep_][m.named_steps['guard'].keep_]
    coef = m.named_steps['clf'].coef_[0]
    nz = set(kept[coef[:len(kept)] != 0])
    ind = m.named_steps['impute'].indicator_
    if ind is not None:  # missing-indicator columns -> parent variable
        nz |= set(kept[ind.features_][coef[len(kept):] != 0])
    return nz


def stability_select(Xtr, ytr, names):
    """As expanded_features.stability (same subsample draws), fits in parallel; C chosen on this training set only."""
    m = l1_model().fit(Xtr, ytr)
    rng = np.random.RandomState(SEED)
    subs = [np.concatenate([rng.choice(np.where(ytr == k)[0], (ytr == k).sum() // 2, replace=False) for k in (0, 1)])
            for _ in range(N_SUB)]
    sets = Parallel(n_jobs=-1)(delayed(_nonzero)(fixed(m), Xtr[i], ytr[i], names) for i in subs)
    freq = pd.Series(0.0, index=names)
    for s in sets:
        freq[list(s)] += 1 / N_SUB
    freq = freq[freq >= THRESH].sort_values(ascending=False)
    return {'C': float(m.named_steps['clf'].C_[0]), 'selected': {v: float(f) for v, f in freq.items()}}


def stage_select():
    tiers, labels, _ = inventory()
    X, y, itr, ite, _ = load(tiers)
    cols = [c for c in tiers[TIER] if c in X.columns]
    Xtr, ytr = X.iloc[itr][cols].values, y[itr]
    out = {'tier': TIER, 'threshold': THRESH, 'n_subsamples': N_SUB, 'folds': []}
    for k, (a, b) in enumerate(CV.split(Xtr, ytr)):  # split-unit:ok one row per participant (asserted in load)
        assert not set(a) & set(b), 'unit leak'
        t = time.time()
        r = stability_select(Xtr[a], ytr[a], cols)
        log(f'fold {k}: train {len(a)} participants / {ytr[a].sum()} events, val {len(b)} / {ytr[b].sum()}; '
            f'{len(r["selected"])} selected, {time.time() - t:.0f}s')
        out['folds'].append(r)
    out['full_train'] = stability_select(Xtr, ytr, cols)
    out['labels'] = {v: labels.get(v, '')[:110] for r in out['folds'] + [out['full_train']] for v in r['selected']}
    log(f'full train: {len(out["full_train"]["selected"])} selected: {list(out["full_train"]["selected"])}')
    json.dump(out, open(SEL, 'w'), indent=1)


def stage_fit():
    from autogluon.tabular import TabularPredictor
    sel = json.load(open(SEL))
    tiers, _, _ = inventory()
    X, y, itr, ite, ref = load(tiers)
    Xtr, Xte, ytr, yte = X.iloc[itr].reset_index(drop=True), X.iloc[ite].reset_index(drop=True), y[itr], y[ite]
    run_dir = os.path.join(ROOT, 'data', f"autogluon_expanded_{time.strftime('%Y%m%d_%H%M%S')}")

    def fit_ag(A, yy, tag):
        return TabularPredictor('y', eval_metric='roc_auc', path=os.path.join(run_dir, tag), verbosity=1).fit(
            A.assign(y=yy), presets='best_quality', time_limit=AG_S)

    p_oof, folds = np.zeros(len(ytr)), []
    for k, (a, b) in enumerate(CV.split(Xtr, ytr)):  # split-unit:ok one row per participant (asserted in load)
        assert not set(a) & set(b), 'unit leak'
        folds.append({'n_train_units': len(a), 'n_test_units': len(b), 'n_overlap_units': 0})
        v = list(sel['folds'][k]['selected'])
        p_oof[b] = fit_ag(Xtr.loc[a, v], ytr[a], f'fold{k}').predict_proba(Xtr.loc[b, v])[1].values
        log(f'fold {k}: {len(v)} variables, fold AUC {roc_auc_score(ytr[b], p_oof[b]):.3f}')
    v = list(sel['full_train']['selected'])
    ag = fit_ag(Xtr[v], ytr, 'full')
    p = ag.predict_proba(Xte[v])[1].values
    l1 = pd.read_csv(os.path.join(OUT, 'predictions_test.csv'))[f'{TIER}__L1_LR'].values
    res = {'_note': 'EXPLORATORY, POST HOC (chosen after expanded-predictor results were visible). '
                    'Stability selection refitted inside each outer fold and on the full training set. '
                    'Source: src/expanded_autogluon.py',
           'script': 'src/expanded_autogluon.py', 'git_hash': GIT, 'unit_of_analysis': 'participant (PublicID)',
           'splitter': 'StratifiedKFold(5) over participant rows; test = expanded_features.load() split', 'groups_passed': False,
           'folds': folds, 'n_train_units': len(ytr), 'n_test_units': len(yte),
           'tier': TIER, 'preset': 'best_quality', 'time_limit_s': AG_S, 'variables_full_train': v,
           'n_variables_per_fold': [len(f['selected']) for f in sel['folds']],
           'best_model': ag.model_best, 'oof_roc_auc': float(roc_auc_score(ytr, p_oof)),
           'test': summarise(yte, p, {'default_0.5': 0.5})}
    for name, q in (('LR_tuned_18var', ref['LR_tuned'].values), (f'{TIER}__L1_LR', l1)):
        d, z, pv = delong(yte, p, q)
        res[f'delong_vs_{name}'] = {'diff': d, 'z': z, 'p': pv}
    json.dump(res, open(os.path.join(OUT, 'autogluon.json'), 'w'), indent=1, default=float)
    pd.DataFrame({'AutoGluon_tier2_stab': p}).to_csv(os.path.join(OUT, 'predictions_test_autogluon.csv'), index=False)
    pd.DataFrame({'AutoGluon_tier2_stab': p_oof}).to_csv(os.path.join(OUT, 'predictions_train_oof_autogluon.csv'), index=False)
    log(f'AutoGluon: OOF {res["oof_roc_auc"]:.3f}, test {res["test"]["roc_auc"]:.3f}')


if __name__ == '__main__':
    {'select': stage_select, 'fit': stage_fit}[sys.argv[1]]()
