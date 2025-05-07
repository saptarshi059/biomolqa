from pathlib import Path
import pandas as pd

drug_df = pd.read_csv("../../data/mined_data/DDI_subset.csv")

drug_id_name = {}
sample_id = 0
for row in drug_df.itertuples():
    if (row.drug_1_name, row.drug_1_SMILES) not in drug_id_name:
        drug_id_name[(row.drug_1_name, row.drug_1_SMILES)] = sample_id
        sample_id += 1
    
    if (row.drug_2_name, row.drug_2_SMILES) not in drug_id_name:
        drug_id_name[(row.drug_2_name, row.drug_2_SMILES)] = sample_id
        sample_id += 1

for key, val in drug_id_name.items():
    with Path(f"../../data/background_information_data/drug_data/SMILES/{val}.txt").open("w") as file:
        file.write(f"DRUG NAME: {key[0]}\nDRUG SMILES: {key[1]}")