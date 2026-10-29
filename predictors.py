#!/usr/bin/env python3
"""
Predictor variables for nuMoM2b postpartum risk prediction analysis.

This module defines the list of predictor variables used in the AutoML pipeline
for predicting maternal postpartum rehospitalization risk.
"""

# List of predictor variables for the analysis
predictors = [
    "CRace",
    "Race", 
    "eRace",
    "eHispanic",
    "BMI",
    "BMI_Cat",
    "Education",
    "GravCat",
    "SmokeCat1",
    "SmokeCat2",
    "SmokeCat3",
    "Ins_Govt",
    "Ins_Mil",
    "Ins_Comm",
    "Ins_Pers",
    "Ins_Othr",
    "PctFedPoverty",
    "poverty"
]

if __name__ == "__main__":
    print("Predictor variables for nuMoM2b analysis:")
    print("-" * 40)
    for i, predictor in enumerate(predictors, 1):
        print(f"{i:2d}. {predictor}")
    print(f"\nTotal number of predictors: {len(predictors)}")