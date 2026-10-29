"""Internal (framework-reported) vs test ROC-AUC for the AutoML runs.

FLAML stores best_loss = 1 - ROC-AUC on its own validation data; this converts it so the
manuscript can cite the internal AUC directly. Writes results/revision/automl_internal.json.
"""
import json, os

OUT = os.environ.get('REVISION_OUT', 'results/revision')
models = json.load(open(f'{OUT}/models.json'))['models']
ev = json.load(open(f'{OUT}/evaluate.json'))['models']
out: dict = {name: {'internal_roc_auc': round(1 - models[name]['best_cv_loss'], 4),
              'validation': 'holdout on SMOTE-resampled training data' if models[name]['smote']
                            else '5-fold CV on original training data',
              'test_roc_auc': round(ev[name]['roc_auc'], 4)}
       for name in ('FLAML_default', 'FLAML_tuned')}
out['_provenance'] = 'src/automl_internal.py from models.json (best_cv_loss) and evaluate.json'
json.dump(out, open(f'{OUT}/automl_internal.json', 'w'), indent=2)
print(json.dumps(out, indent=2))
