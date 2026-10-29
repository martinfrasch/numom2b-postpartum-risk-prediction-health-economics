# Limits of sociodemographic data for predicting postpartum readmission

Analysis code and aggregate results for:

> Crabtree L, Wakefield C, Gheorghe CP, Frasch MG. *Limits of Sociodemographic Data for Predicting Postpartum Readmission: Conventional, Ensemble and Automated Machine Learning Compared at Equal Workload, With a Decision-Economic Analysis.* Preprint: [medRxiv 10.1101/2025.12.16.25342440](https://doi.org/10.1101/2025.12.16.25342440).

Interactive explorer of the economic analysis: <https://martinfrasch.github.io/postpartum-readmission-economics/> ([doi:10.5281/zenodo.23749950](https://doi.org/10.5281/zenodo.23749950)).

## Data

The study uses the nuMoM2b cohort (first pregnancies, 8 US centres). Participant-level data are held by the NICHD Data and Specimen Hub (DASH, <https://dash.nichd.nih.gov>) and are available to qualified investigators under a data use agreement. **No data, predictions or result files are included here**, only code. The scripts write aggregate result files under `results/`, suppressing any count below 11.

To reproduce, place the DASH files in `data/` (or set `NUMOM_RAW` and `NUMOM_CODEBOOK`):

- `data/nuMoM2b_Dataset_NICHD_Data_Challenge.csv`
- `data/nuMoM2b_Codebook_NICHD Data Challenge.xlsx` (only for the post hoc variable search)

Public inputs: CDC WONDER natality marginals (`external/wonder/`, included) and the T-MSIS Medicaid provider spending files (`MEDICAID_DIR`, not included; used only by `src/medicaid_contact_costs.py`).

## Environment

Python 3.11 with the pinned packages in `requirements-revision.txt`:

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-revision.txt
```

## Reproducing the results

Unit of analysis: the participant (one row each). The 70/30 train/test split is stratified on the outcome with a fixed seed; every reporting script asserts that no participant is in both sets, and result files record the split, participants per set and overlap (`_provenance`).

| Step | Command | Writes |
|---|---|---|
| Analysis input | `python src/make_analysis_subset.py` | `data/numom2b_analysis_subset.csv` |
| Models, AutoML, evaluation, data limitation | `python src/revision_pipeline.py` | `results/revision/{models,autogluon,evaluate,datalimit}.json` |
| Cohort table | `python src/cohort_table.py` | `results/revision/cohort_table.json`, Table 1 |
| AutoML internal validation | `python src/automl_internal.py` | `results/revision/automl_internal.json` |
| Figures 1–4, workload, decision curves | `python src/generate_revision_figures.py` | `results/manuscript/figures/`, `workload.json`, `dca_summary.json` |
| Economic analysis | `python src/economic_check.py` | `results/revision/economic_check.json` |
| Figures 5–6 | `python src/figure_economic_window.py`, `python src/figure_economic_forest.py` | `results/manuscript/figures/` |
| Post hoc: variables recorded before discharge | `python src/expanded_features.py`, then `expanded_autogluon.py`, `expanded_economics.py`, `expanded_prevalence_economics.py`, `table_s6_stability.py` | `results/expanded/`, Table S6 |
| Post hoc: Medicaid prices, payer and state transport | `python src/medicaid_contact_costs.py`, `subgroup_payer_economics.py`, `regional_economics.py`, `state_reweighted_economics.py`, `figure_state_transport.py` | `results/expanded/`, Figure S1 |
| Explorer data | `python src/public_bins.py` | `results/revision/economic_bins_public.json` (explorer page: [postpartum-readmission-economics](https://github.com/martinfrasch/postpartum-readmission-economics)) |

`src/economic_check.py` also writes `results/revision/economic_bins.json`, risk bins of 10 patients with small cells; do not publish it. `src/public_bins.py` writes the version used by the interactive explorer, with at least 11 readmissions and 11 non-readmissions per bin.

Several steps (AutoML in particular, and the L1 solver in the post hoc search) are not bit-for-bit deterministic; reruns agree to the precision reported in the paper.

## Provenance

Each result file records the script that wrote it and the git commit (`git_hash`). These hashes refer to the authors' development repository, from which this repository is a cleaned snapshot.

## Licence

MIT (see `LICENSE`).
