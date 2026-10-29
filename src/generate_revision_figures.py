"""Figures 1-4 and Supplementary Table S5 for the revised manuscript.
Reads only results/revision/*.json and the local prediction files written by src/revision_pipeline.py
(no metric literals). Outputs to results/manuscript/figures/ and results/manuscript/tables/."""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.revision_pipeline import OUT as R, SEARCH, recalibrate  # R honours REVISION_OUT

FIG = os.environ.get('FIG_OUT') or os.path.join(ROOT, 'results', 'manuscript', 'figures')
TAB = os.environ.get('FIG_OUT') or os.path.join(ROOT, 'results', 'manuscript', 'tables')
ev, mo, dl = (json.load(open(os.path.join(R, f))) for f in ('evaluate.json', 'models.json', 'datalimit.json'))
BLUE, ORANGE, AQUA, INK2, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#898781', '#e1e0d9'
NAMES = [('LR_orig', 'LR, as submitted'), ('LR_orig_noSMOTE', 'LR, as submitted, no SMOTE'), ('LR_tuned', 'LR, tuned'),
         ('RF_orig', 'RF, as submitted'), ('RF_tuned', 'RF, tuned'), ('GB_orig', 'GB, as submitted'), ('GB_tuned', 'GB, tuned'),
         ('Stacking_orig', 'Stacking (submitted bases)'), ('Stacking_tuned', 'Stacking (tuned bases)'),
         ('SoftVote_orig', 'Soft voting (submitted bases)'), ('SoftVote_tuned', 'Soft voting (tuned bases)'),
         ('FLAML_default', 'FLAML, as submitted'), ('FLAML_tuned', 'FLAML, imbalance-aware'), ('AutoGluon', 'AutoGluon')]
BREAKS = {3, 7, 11}
ys, y = [], 0
for i in range(len(NAMES)):
    if i in BREAKS: y += 0.6
    ys.append(y); y += 1
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf')); fig.savefig(os.path.join(FIG, name + '.png'), dpi=200); plt.close(fig)
    print('wrote', name)


# ---- Figure 1: test AUC (95% CI) and cross-validated training AUC for every model
fig, ax = plt.subplots(figsize=(7, 5.2))
ax.axvline(0.5, color=MUTED, lw=1, ls='--')
for (k, n), yy in zip(NAMES, ys):
    m = ev['models'][k]; lo, hi = m['ci95']['roc_auc']
    ax.plot([lo, hi], [yy, yy], color=BLUE, lw=2, solid_capstyle='round')
    ax.plot(m['roc_auc'], yy, 'o', color=BLUE, ms=6, mec='white', mew=1.2, zorder=3)
    ax.plot(m['oof_auc'], yy, 'D', color=ORANGE, ms=5, mfc='none', mew=1.5, zorder=3)
ax.set_yticks(ys); ax.set_yticklabels([n for _, n in NAMES]); ax.invert_yaxis()
ax.set_xlabel('ROC-AUC'); ax.grid(axis='x', color=GRID)
ax.legend(handles=[Line2D([], [], color=BLUE, marker='o', lw=2, label='Test set (95% bootstrap CI)'),
                   Line2D([], [], color=ORANGE, marker='D', mfc='none', lw=0, label='Cross-validated, training set'),
                   Line2D([], [], color=MUTED, ls='--', label='Chance')], loc='lower center', bbox_to_anchor=(0.4, 1.0), ncol=3, frameon=False, fontsize=8)
fig.tight_layout(); save(fig, 'figure1_auc_all_models')

# ---- Figure 2: sensitivity at the default 0.5 threshold vs at a matched, training-selected flag rate
fl_key = next(k for k in ev['models']['LR_tuned']['operating_points'] if k.startswith('testtop_flag'))
fig, ax = plt.subplots(figsize=(7, 5.2))
for (k, n), yy in zip(NAMES, ys):
    op = ev['models'][k]['operating_points']; a = op['default_0.5']['sensitivity'] * 100; b = op[fl_key]['sensitivity'] * 100
    lo, hi = (v * 100 for v in ev['models'][k]['ci95']['sens@' + fl_key])
    ax.plot([lo, hi], [yy + 0.12, yy + 0.12], color=BLUE, lw=2, alpha=0.9)
    ax.plot(b, yy + 0.12, 'o', color=BLUE, ms=6, mec='white', mew=1.2, zorder=3)
    ax.plot(a, yy - 0.12, 's', color=INK2, ms=5, zorder=3)
ax.set_yticks(ys); ax.set_yticklabels([n for _, n in NAMES]); ax.invert_yaxis()
pct = ev['test_flag_rate_reference'] * 100
ax.set_xlabel('Sensitivity on the test set (%)'); ax.set_xlim(-2, 100); ax.grid(axis='x', color=GRID)
ax.legend(handles=[Line2D([], [], color=INK2, marker='s', lw=0, label='Default threshold 0.5'),
                   Line2D([], [], color=BLUE, marker='o', lw=2, label=f'Top {pct:.1f}% of test patients flagged by every model (95% CI)')],
          loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8)
fig.tight_layout(); save(fig, 'figure2_sensitivity_default_vs_matched')

# ---- Figure 3: (a) calibration before/after Platt recalibration, (b) decision curves after recalibration
te = pd.read_csv(os.path.join(R, 'predictions_test.csv')); tr = pd.read_csv(os.path.join(R, 'predictions_train_oof.csv'))
te = te.join(pd.read_csv(os.path.join(R, 'predictions_test_autogluon.csv')))
tr = tr.join(pd.read_csv(os.path.join(R, 'predictions_train_oof_autogluon.csv')))
yt, yo = te['y'].values, tr['y'].values
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.8))


def cal_points(p, n=5):
    q = np.quantile(p, np.linspace(0, 1, n + 1)); idx = np.clip(np.searchsorted(q, p, side='right') - 1, 0, n - 1)
    return [p[idx == i].mean() * 100 for i in range(n)], [yt[idx == i].mean() * 100 for i in range(n)]


for k, col, lab in (('LR_orig', ORANGE, 'LR as submitted (SMOTE, class weights)'), ('LR_tuned', BLUE, 'LR tuned')):
    x, o = cal_points(te[k].values); a1.plot(x, o, 'o-', color=col, lw=1.5, ms=5, label=lab + ', raw')
    pc = recalibrate(yo, tr[k].values, te[k].values)['platt']; x, o = cal_points(pc)
    a1.plot(x, o, 'o--', color=col, lw=1.5, ms=5, mfc='white', label=lab + ', Platt-recalibrated')
a1.axhline(yt.mean() * 100, color=MUTED, lw=1, ls=':', label='Observed readmission rate')
a1.set_xscale('log'); a1.set_xlabel('Mean predicted risk (%, log scale)'); a1.set_ylabel('Observed readmissions (%)')
a1.set_title('a  Calibration (test set, risk quintiles)', loc='left', fontsize=9.5)
a1.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
a1.grid(color=GRID)
ts = np.array(ev['dca_thresholds']) * 100
a2.plot(ts, ev['dca_treat_all'], color=ORANGE, lw=2, label='Contact all')
a2.axhline(0, color=MUTED, lw=1, label='Contact none')
for k, col, ls in (('LR_tuned', BLUE, '-'), ('GB_tuned', AQUA, '--'), ('Stacking_orig', INK2, ':')):
    a2.plot(ts, ev['models'][k]['recalibrated']['platt']['net_benefit'], color=col, lw=1.8, ls=ls, label=dict(NAMES)[k] + ' (recalibrated)')
a2.set_ylim(-0.003, max(ev['dca_treat_all']) * 1.1); a2.set_xlim(ts[0], ts[-1])
a2.set_xlabel('Threshold probability (%)'); a2.set_ylabel('Net benefit')
a2.set_title('b  Decision curves (test set)', loc='left', fontsize=9.5); a2.grid(color=GRID)
a2.legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.tight_layout(); save(fig, 'figure3_calibration_dca')

# ---- Figure 4: data limitation. (a) learning curve, (b) test AUC by feature set
lc = dl['learning_curve_LR']; fig, (b1, b2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1, 1.3]})
for key, col, lab in (('learning_curve_LR', INK2, 'Pre-specified: LR with RFE (10 features)'),
                      ('learning_curve_LR_tuned_noRFE_posthoc', BLUE, 'Post hoc: tuned LR, all 18 variables')):
    c = dl[key]; m, s, n = np.array(c['auc_mean']), np.array(c['auc_sd']), np.array(c['train_sizes'])
    b1.fill_between(n, m - s, m + s, color=col, alpha=0.12, lw=0); b1.plot(n, m, 'o-', color=col, lw=2, label=lab)
b1.legend(fontsize=7.5, frameon=False, loc='lower right')
b1.axhline(0.5, color=MUTED, lw=1, ls='--'); b1.set_xlabel('Training participants'); b1.set_ylabel('Cross-validated ROC-AUC (mean ± SD)')
b1.set_title('a  Learning curves (sociodemographic, training-set CV)', loc='left', fontsize=9.5); b1.grid(color=GRID)
SETS = [('sociodemographic', 'Socio-\ndemographic'), ('enrollment_extended', '+ enrolment\ninterview'),
        ('discharge_available', '+ delivery and\ndischarge'), ('sensitivity_discharge_plus_untimed_hypertension', '+ untimed\nhypertension\n(sensitivity)')]
for j, (mk, col, off) in enumerate((('LR', BLUE, -0.12), ('GB', AQUA, 0.12))):
    for i, (k, lab) in enumerate(SETS):
        r = dl[k][mk]; lo, hi = r['ci95']['roc_auc']
        b2.plot([i + off, i + off], [lo, hi], color=col, lw=2); b2.plot(i + off, r['test_auc'], 'o', color=col, ms=6, mec='white', mew=1.2,
                                                                       label=('Logistic regression' if mk == 'LR' else 'Gradient boosting') if i == 0 else None)
b2.axhline(0.5, color=MUTED, lw=1, ls='--'); b2.set_xticks(range(len(SETS))); b2.set_xticklabels([l for _, l in SETS], fontsize=8)
b2.set_ylabel('Test ROC-AUC (95% CI)'); b2.set_title('b  Feature sets (tuned models)', loc='left', fontsize=9.5)
b2.legend(fontsize=8, frameon=False, loc='upper left'); b2.grid(axis='y', color=GRID)
fig.tight_layout(); save(fig, 'figure4_data_limitation')

# ---- Supplementary Table S5: search spaces and selected hyperparameters
def fmt(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        vals = list(v)
        if all(isinstance(x, (float, np.floating)) for x in vals) and len(vals) > 6:
            return f'{len(vals)} log-spaced values, {min(vals):.0e}--{max(vals):.0e}'
        return ', '.join('SMOTE' if not isinstance(x, (str, int, float, type(None))) else ('none' if x == 'passthrough' else str(x)) for x in vals)
    return str(v)


lines = ['% src: src/revision_pipeline.py (SEARCH) and results/revision/models.json (generated by src/generate_revision_figures.py)',
         '\\begin{tabular}{llL{6.2cm}l}', '\\toprule', '\\textbf{Model} & \\textbf{Parameter} & \\textbf{Search space} & \\textbf{Selected} \\\\', '\\midrule']
for name, (_, grid, n_iter) in SEARCH.items():
    best = mo['models'][name + '_tuned']['best_params']
    for i, (p, space) in enumerate(grid.items()):
        sel = best.get(p); sel = f'{sel:.4g}' if isinstance(sel, float) else ('none' if sel == 'passthrough' else str(sel))
        lines.append(f"{name if i == 0 else ''} & \\texttt{{{p.replace('clf__', '').replace('_', chr(92) + '_')}}} & "
                     f"{fmt(space).replace('_', chr(92) + '_')} & {sel.replace('_', chr(92) + '_')} \\\\")
    lines.append(f"& \\multicolumn{{3}}{{l}}{{\\footnotesize Randomised search, {n_iter} draws, stratified 5-fold CV on the training set, ROC-AUC; "
                 f"CV AUC of selected {mo['models'][name + '_tuned']['cv_auc']:.3f}}} \\\\ \\midrule")
lines[-1] = lines[-1].replace(' \\midrule', '')
lines += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(TAB, 'tableS5_hyperparameters.tex'), 'w').write('\n'.join(lines) + '\n'); print('wrote tableS5')

# ---- Supplementary Table S6: feature-set definitions (from the pipeline's own lists)
from src.revision_pipeline import CANDIDATES, ENROLLMENT_EXTRA, DISCHARGE_EXTRA, HYPERTENSION_UNTIMED
LAB = {'CRace': 'race/ethnicity (5 groups)', 'Race': 'race/ethnicity (8 groups)', 'eRace': 'race (NIH categories)', 'eHispanic': 'Hispanic ethnicity',
       'BMI': 'BMI at enrolment', 'BMI_Cat': 'BMI category', 'Education': 'education', 'GravCat': 'gravidity category',
       'SmokeCat1': 'ever used tobacco', 'SmokeCat2': 'smoked in 3 months before pregnancy', 'SmokeCat3': 'cigarettes per day (none if not smoking)',
       'Ins_Govt': 'government insurance', 'Ins_Mil': 'military insurance', 'Ins_Comm': 'commercial insurance', 'Ins_Pers': 'paid by personal income',
       'Ins_Othr': 'other payer', 'PctFedPoverty': 'income, \\% of federal poverty level', 'poverty': 'poverty category',
       'Age_at_V1': 'age', 'AgeCat_V1': 'age category', 'V1AF04': 'marital status', 'V1AD15': 'asthma', 'V1AE01': 'number of pregnancies',
       'V1AG04': 'ever used tobacco (interview)', 'V1AG05': 'smoked in 3 months before pregnancy (interview)',
       'V1AD02a': 'counselled: folic acid', 'V1AD02b': 'counselled: healthy weight', 'V1AD02c': 'counselled: vaccines',
       'V1AD02d': 'counselled: dental visit', 'V1AD02e': 'counselled: genetic counselling',
       'CMAC15': 'cesarean delivery', 'CMAC12': 'operative vaginal delivery attempted', 'CMAE03': 'diabetes ever diagnosed',
       'CMAE04a1a': 'treated for depression before pregnancy', 'CMAE04a1b': 'treated for depression during pregnancy',
       'CMAE04a2a': 'treated for anxiety before pregnancy', 'CMAE04a2b': 'treated for anxiety during pregnancy',
       'A09A03a': 'preterm birth ($<$37 weeks)', 'SGA_alex': 'small for gestational age',
       'CMAE01': 'hypertension, proteinuria or related condition', 'A09A03b': 'suspected maternal hypertensive disorder'}
rows = [('Sociodemographic (primary)', CANDIDATES), ('Added: enrolment interview', ENROLLMENT_EXTRA),
        ('Added: delivery and discharge', DISCHARGE_EXTRA), ('Added in sensitivity analysis only (timing not recorded)', HYPERTENSION_UNTIMED)]
t = ['% generated by src/generate_revision_figures.py from src/revision_pipeline.py', '\\begin{tabular}{L{4.2cm}L{11cm}}', '\\toprule',
     '\\textbf{Set} & \\textbf{Variables (nuMoM2b name: description)} \\\\', '\\midrule']
for name, vs in rows:
    t.append(f"{name} ({len(vs)}) & " + '; '.join(f"\\texttt{{{v.replace('_', chr(92) + '_')}}}: {LAB[v]}" for v in vs) + ' \\\\')
t += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(TAB, 'tableS6_feature_sets.tex'), 'w').write('\n'.join(t) + '\n'); print('wrote tableS6')

# ---- Table 2: all models on the test set (generated)
def p3(x): return f'{x:.3f}'
tk = next(k for k in ev['models']['LR_tuned']['operating_points'] if k.startswith('testtop_flag'))
t = ['% src: results/revision/evaluate.json (generated by src/generate_revision_figures.py; do not edit by hand)',
     '\\begin{tabular}{lcccccccc}', '\\toprule',
     '& \\textbf{CV AUC} & \\multicolumn{2}{c}{\\textbf{Test ROC-AUC}} & \\textbf{PR-AUC} & \\multicolumn{2}{c}{\\textbf{Brier}} & '
     '\\textbf{Sens. at 0.5} & \\textbf{Sens. at equal} \\\\',
     '\\textbf{Model} & \\textbf{(train)} & \\textbf{Est.} & \\textbf{95\\% CI} & & \\textbf{Raw} & \\textbf{Recal.} & '
     '\\textbf{(\\% flagged)} & \\textbf{workload$^a$ (95\\% CI)} \\\\', '\\midrule']
for i, (k, n) in enumerate(NAMES):
    if i in BREAKS: t.append('\\addlinespace')
    m = ev['models'][k]; d5 = m['operating_points']['default_0.5']; eq = m['operating_points'][tk]; ci = m['ci95']
    t.append(f"{n} & {p3(m['oof_auc'])} & {p3(m['roc_auc'])} & {p3(ci['roc_auc'][0])}--{p3(ci['roc_auc'][1])} & {p3(m['pr_auc'])} & "
             f"{m['brier']:.4f} & {m['recalibrated']['platt']['brier']:.4f} & {d5['sensitivity']*100:.1f} ({d5['pct_flagged']*100:.1f}) & "
             f"{eq['tp']}/46, {eq['sensitivity']*100:.1f} ({ci['sens@'+tk][0]*100:.1f}--{ci['sens@'+tk][1]*100:.1f}) \\\\")
t += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(TAB, 'table2_models.tex'), 'w').write('\n'.join(t) + '\n'); print('wrote table2')

# ---- Table 3: data limitation (generated)
t = ['% src: results/revision/datalimit.json (generated by src/generate_revision_figures.py; do not edit by hand)',
     '\\begin{tabular}{llccccc}', '\\toprule',
     '\\textbf{Feature set} & \\textbf{Model} & \\textbf{Variables} & \\textbf{CV AUC} & \\textbf{Test ROC-AUC (95\\% CI)} & \\textbf{Test PR-AUC (95\\% CI)} & \\textbf{Brier} \\\\', '\\midrule']
for k, lab in SETS:
    for mk, mn in (('LR', 'Logistic regression'), ('GB', 'Gradient boosting')):
        r = dl[k][mk]
        t.append(f"{lab.replace(chr(10), ' ') if mk == 'LR' else ''} & {mn} & {len(dl[k]['features']) if mk == 'LR' else ''} & {p3(r['cv_auc'])} & "
                 f"{p3(r['test_auc'])} ({p3(r['ci95']['roc_auc'][0])}--{p3(r['ci95']['roc_auc'][1])}) & "
                 f"{p3(r['test_pr_auc'])} ({p3(r['ci95']['pr_auc'][0])}--{p3(r['ci95']['pr_auc'][1])}) & {r['test_brier']:.4f} \\\\")
t += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(TAB, 'table3_data_limitation.tex'), 'w').write('\n'.join(t) + '\n'); print('wrote table3')

# ---- Workload per 1,000 deliveries at each operating point (Referee 3.4); written as a result file
work = {}
for k in ('LR_orig', 'LR_tuned', 'SoftVote_tuned', 'AutoGluon'):
    for op_name, op in ev['models'][k]['operating_points'].items():
        if op['tp'] + op['fp'] == 0: continue
        work.setdefault(k, {})[op_name] = {'flagged_per_1000': op['pct_flagged'] * 1000,
                                           'readmissions_reached_per_1000': op['tp'] / (op['tp'] + op['fp'] + op['tn'] + op['fn']) * 1000,
                                           'readmissions_missed_per_1000': op['fn'] / (op['tp'] + op['fp'] + op['tn'] + op['fn']) * 1000,
                                           'number_needed_to_screen': (op['tp'] + op['fp']) / op['tp'] if op['tp'] else None,
                                           'sensitivity': op['sensitivity'], 'threshold': op['threshold']}
import subprocess as _sp
_git = _sp.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip() + \
    ('+uncommitted' if _sp.run(['git', 'status', '--porcelain', __file__], capture_output=True, text=True).stdout.strip() else '')
PROV = {'script': 'src/generate_revision_figures.py', 'git_hash': _git, 'unit_of_analysis': 'participant',
        'derived_from': 'results/revision/evaluate.json (split provenance recorded there)'}
work['_provenance'] = PROV
json.dump(work, open(os.path.join(R, 'workload.json'), 'w'), indent=1); print('wrote workload.json')

# ---- Decision-curve summary: largest net-benefit gain over the better of contact-all / contact-none (recalibrated)
dca = {}
for k in ev['models']:
    for meth in ('platt', 'isotonic'):
        nb = ev['models'][k]['recalibrated'][meth]['net_benefit']
        g = [a - max(b, 0) for a, b in zip(nb, ev['dca_treat_all'])]
        i = int(np.argmax(g))
        dca.setdefault(k, {})[meth] = {'max_gain_net_benefit': g[i], 'max_gain_per_1000': g[i] * 1000,
                                       'at_threshold': ev['dca_thresholds'][i]}
dca['_overall_max_platt'] = max(v['platt']['max_gain_net_benefit'] for k, v in dca.items() if not k.startswith('_'))
dca['_provenance'] = PROV
json.dump(dca, open(os.path.join(R, 'dca_summary.json'), 'w'), indent=1); print('wrote dca_summary.json')
