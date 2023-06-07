from openbabel import pybel
import pandas as pd

from src.utils.constants import ID, SMILE
from src.utils.constants import ACTIVE, INACTIVE, UNKNOWN
from src.utils.constants import ASSAYS, ASSAYS_LOWER, ID_SDF
from src.utils.constants import EMPTY_ACT_LIST

# Process data from the original Tox21 datasets
# 1. Obtain smiles
smiles_dict = {}
smiles_file = "../data/tox21/leader/orig/data.smiles"
smiles = pd.read_csv(smiles_file, sep="\t", header=None, names=[SMILE, ID])

for _, row in smiles.iterrows():
    id = row[ID]
    smile = row[SMILE]

    if id not in smiles_dict:
        smiles_dict[id] = smile
    else:
        raise ValueError("Repeated compound.")

# Save smiles_df
smiles_df = pd.DataFrame(smiles_dict.items())
smiles_df.columns = [ID, SMILE]
smiles_df.set_index(ID, inplace=True)
smiles_df.to_csv("../data/tox21/leader/processed/smiles.csv", sep=",", header=True, index=True)

# 2. Obtain activities
activity_df = pd.DataFrame(columns=[ID] + ASSAYS_LOWER)
activity_df.set_index(ID, inplace=True)

activity_file = "../data/tox21/leader/orig/data.sdf"
for mol in pybel.readfile("sdf", activity_file):
    id = mol.data[ID_SDF]

    if id not in smiles_dict:
        raise ValueError("Unknown compound.")
    if id in activity_df.index:
        raise ValueError("Repeated compound.")

    activity_df.loc[id] = EMPTY_ACT_LIST.copy()
    for assay in ASSAYS:
        if assay in mol.data.keys():
            activity = mol.data[assay]
        else:
            activity = UNKNOWN
        activity_df.loc[id, assay.lower()] = activity

# Save activity_df
activity_df.to_csv("../data/tox21/leader/processed/activity.csv", sep=",", header=True, index=True)

# 3. Save activities and smiles df
smiles_activity_df = smiles_df.join(activity_df)
smiles_activity_df.to_csv("../data/tox21/leader/processed/smiles_activity.csv", sep=",", header=True, index=True)
