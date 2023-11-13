import os
import pandas as pd

import src.utils.helpers as helpers

# Activity df
train_activity_df = helpers.read_df("../data/tox21/train/processed/activity.csv")
leader_activity_df = helpers.read_df("../data/tox21/leader/processed/activity.csv")
eval_activity_df = helpers.read_df("../data/tox21/eval/processed/activity.csv")

activity_df = pd.concat([train_activity_df, leader_activity_df, eval_activity_df])
helpers.write_df(activity_df, "../data/tox21/full/activity.csv")

# Smiles df
train_smiles_df = helpers.read_df("../data/tox21/train/processed/smiles.csv")
leader_smiles_df = helpers.read_df("../data/tox21/leader/processed/smiles.csv")
eval_smiles_df = helpers.read_df("../data/tox21/eval/processed/smiles.csv")

smiles_df = pd.concat([train_smiles_df, leader_smiles_df, eval_smiles_df])
helpers.write_df(smiles_df, "../data/tox21/full/smiles.csv")

# Ids
train_smiles_df.to_csv('../data/tox21/full/train_ids.csv', columns=[])
leader_smiles_df.to_csv('../data/tox21/full/leader_ids.csv', columns=[])
eval_smiles_df.to_csv('../data/tox21/full/eval_ids.csv', columns=[])
