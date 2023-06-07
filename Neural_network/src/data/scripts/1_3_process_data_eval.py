from openbabel import pybel
import pandas as pd

from src.utils.constants import ID, SMILE
from src.utils.constants import ACTIVE, INACTIVE, UNKNOWN
from src.utils.constants import ASSAYS, ASSAYS_LOWER, ID_SDF
from src.utils.constants import EMPTY_ACT_LIST

# Process data from the original Tox21 datasets
# 1. Obtain smiles
smiles_dict = {}
smiles_file = "../data/tox21/eval/orig/data.smiles"
smiles = pd.read_csv(smiles_file, sep="\t", header=0, names=[SMILE, ID])

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
smiles_df.to_csv("../data/tox21/eval/processed/smiles.csv", sep=",", header=True, index=True)

# 2. Obtain activities
activity_file = "../data/tox21/eval/orig/labels.txt"
activity_df = pd.read_csv(activity_file, sep="\t", header=0, names=[ID] + ASSAYS_LOWER, index_col=0)

activity_df.replace("x", UNKNOWN, inplace=True)
activity_df.to_csv("../data/tox21/eval/processed/activity.csv", sep=",", header=True, index=True)

# 3. Save activities and smiles df
smiles_activity_df = smiles_df.join(activity_df)
smiles_activity_df.to_csv("../data/tox21/eval/processed/smiles_activity.csv", sep=",", header=True, index=True)
