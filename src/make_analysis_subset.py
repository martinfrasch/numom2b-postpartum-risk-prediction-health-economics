"""Write data/numom2b_analysis_subset.csv (the input of src/revision_pipeline.py and src/cohort_table.py): the columns
listed in COLUMNS, all rows, taken unchanged from the NICHD DASH nuMoM2b file (path: NUMOM_RAW or data/, as in
src/expanded_features.py). One row per participant (PublicID). Not distributed: requires a DASH data use agreement."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get('NUMOM_RAW', os.path.join(ROOT, 'data', 'nuMoM2b_Dataset_NICHD_Data_Challenge.csv'))
OUT = os.path.join(ROOT, 'data', 'numom2b_analysis_subset.csv')
COLUMNS = None  # filled below
COLUMNS = ['PublicID', 'A09A03a', 'A09A03b', 'CMAC12', 'CMAC15', 'CMAE01', 'CMAE03', 'CMAE04a1a', 'CMAE04a1b', 'CMAE04a2a', 'CMAE04a2b', 'CMAJ01', 'Age_at_V1', 'AgeCat_V1', 'CRace', 'Race', 'eRace', 'eHispanic', 'BMI', 'BMI_Cat', 'Education', 'GravCat', 'SmokeCat1', 'SmokeCat2', 'SmokeCat3', 'Ins_Govt', 'Ins_Mil', 'Ins_Comm', 'Ins_Pers', 'Ins_Othr', 'PctFedPoverty', 'poverty', 'SGA_alex', 'V1AD02a', 'V1AD02b', 'V1AD02c', 'V1AD02d', 'V1AD02e', 'V1AD15', 'V1AE01', 'V1AF04', 'V1AG04', 'V1AG05']

if __name__ == '__main__':
    d = pd.read_csv(RAW, usecols=COLUMNS, low_memory=False)[COLUMNS]
    assert d.PublicID.is_unique, 'one row per participant expected'
    d.to_csv(OUT, index=False)
    print('wrote', OUT, d.shape)
