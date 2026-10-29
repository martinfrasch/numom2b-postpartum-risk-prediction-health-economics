"""EXPLORATORY: Medicaid payment per post-discharge contact, nationally and by state, to price the contact cost c
in the decision-economic analysis. Public data only (no nuMoM2b data).

Source: HHS OpenData "Medicaid Provider Spending by HCPCS" (T-MSIS, billing NPI x servicing NPI x HCPCS x month),
billing NPI -> state from the NPPES map built in github.com/martinfrasch/medicaid_JB_PPD_screen_modeling.
Measure: total paid / total claim lines, 2022-01..2023-12 (complete years), rows with TOTAL_PAID > 0 only
(zero rows are mostly denied or capitated encounters). States with < MIN_LINES claim lines for a code group are omitted.
Writes results/expanded/medicaid_contact_costs.json.
"""
import json, os
import duckdb

MED = os.environ.get('MEDICAID_DIR', os.path.expanduser('~/projects/Medicaid provider spending'))
PARQUET, NPI_STATE = f'{MED}/medicaid-provider-spending.parquet', f'{MED}/results/nppes_npi_state.csv'
MIN_LINES = 1000
GROUPS = {  # CPT/HCPCS descriptors (AMA CPT); design constants
    'tcm_moderate_99495': ['99495'], 'tcm_high_99496': ['99496'],
    'phone_physician_99441_3': ['99441', '99442', '99443'], 'phone_nonphysician_98966_8': ['98966', '98967', '98968'],
    'home_visit_est_99347_50': ['99347', '99348', '99349', '99350'], 'postpartum_visit_59430': ['59430'],
}
case = 'CASE ' + ' '.join(f"WHEN HCPCS_CODE IN ({', '.join(repr(c) for c in v)}) THEN '{k}'" for k, v in GROUPS.items()) + ' END'
con = duckdb.connect()
q = f"""
WITH s AS (SELECT CAST(npi AS VARCHAR) npi, state FROM read_csv_auto('{NPI_STATE}', types={{'npi': 'VARCHAR'}})),
d AS (SELECT {case} grp, BILLING_PROVIDER_NPI_NUM npi, TOTAL_PAID paid, TOTAL_CLAIM_LINES lines
      FROM read_parquet('{PARQUET}')
      WHERE HCPCS_CODE IN ({', '.join(repr(c) for v in GROUPS.values() for c in v)})
        AND CLAIM_FROM_MONTH BETWEEN '2022-01' AND '2023-12' AND TOTAL_PAID > 0)
SELECT grp, s.state, SUM(paid) paid, SUM(lines) lines FROM d LEFT JOIN s USING (npi) GROUP BY ALL
"""
df = con.execute(q).df()
out = {'_note': 'Public T-MSIS data; paid per claim line, 2022-2023, TOTAL_PAID > 0. Source: src/medicaid_contact_costs.py',
       'min_lines_per_state': MIN_LINES, 'groups': {}}
for g, d in df.groupby('grp'):
    st = d[d.state.notna() & (d.lines >= MIN_LINES)].assign(per_line=lambda x: x.paid / x.lines).sort_values('per_line')
    out['groups'][g] = {
        'codes': GROUPS[g], 'national_paid_per_line': float(d.paid.sum() / d.lines.sum()), 'national_lines': int(d.lines.sum()),
        'pct_lines_state_unmapped': float(d[d.state.isna()].lines.sum() / d.lines.sum()),
        'n_states': int(len(st)),
        'state_per_line_min_median_max': [float(st.per_line.min()), float(st.per_line.median()), float(st.per_line.max())] if len(st) else None,
        'by_state': {r.state: {'paid_per_line': float(r.per_line), 'lines': int(r.lines)} for r in st.itertuples()}}
    o = out['groups'][g]
    print(f"{g:30s} national ${o['national_paid_per_line']:.2f}/line ({o['national_lines']:,} lines); "
          f"{o['n_states']} states, min/median/max {o['state_per_line_min_median_max']}")
json.dump(out, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'expanded',
                                 'medicaid_contact_costs.json'), 'w'), indent=1)
