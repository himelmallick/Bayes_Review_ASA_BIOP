import os
import pandas as pd

import src.utils.helpers as helpers
from src.utils.constants import ID, SMILE, ACTIVITY
from src.utils.constants import ASSAYS_LOWER
from src.utils.constants import EMPTY_ACT_LIST

# Process data from the original Tox21 datasets
smiles_dict = {}
activity_df = pd.DataFrame(columns=[ID] + ASSAYS_LOWER)
activity_df.set_index(ID, inplace=True)

# NR and SR assays
nr_dir = "../data/tox21/train/orig/nr/smiles/"
sr_dir = "../data/tox21/train/orig/sr/smiles/"
smiles_files = helpers.listdir_fullpath(nr_dir) + helpers.listdir_fullpath(sr_dir)

for smiles_file in smiles_files:
    print(smiles_file)
    assay = smiles_file.split("/")[-1].split(".")[0]
    smiles = pd.read_csv(smiles_file, sep="\t", header=None, names=[SMILE, ID, ACTIVITY])

    # Print progress
    print(assay)
    print(smiles.shape)

    for i, row in smiles.iterrows():
        print(i)
        id = row[ID]
        smile = row[SMILE]
        activity = row[ACTIVITY]

        # Update smiles_dict and check for smiles consistency
        if id not in smiles_dict:
            print("Not in smiles dict.")
            # 1. Store smile in the smiles dictionary
            smiles_dict[id] = smile

            # 2. Initialise line in activity df
            activity_list = EMPTY_ACT_LIST.copy()
            activity_df.loc[id] = activity_list

        else:
            if smiles_dict[id] != smile:
                raise ValueError("Smile strings inconsistent.")

        # 3. Update activity df
        activity_df.loc[id, assay] = activity

# Save smiles_df
smiles_df = pd.DataFrame(smiles_dict.items())
smiles_df.columns = [ID, SMILE]
smiles_df.set_index(ID, inplace=True)
smiles_df.to_csv("../data/tox21/train/processed/smiles.csv", sep=",", header=True, index=True)

# Save activity_df
activity_df.to_csv("../data/tox21/train/processed/activity.csv", sep=",", header=True, index=True)

# Save activities and smiles df
smiles_activity_df = smiles_df.join(activity_df)
smiles_activity_df.to_csv("../data/tox21/train/processed/smiles_activity.csv", sep=",", header=True, index=True)
