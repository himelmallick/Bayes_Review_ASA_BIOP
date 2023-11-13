import src.data.merge as merge
import src.utils.helpers as helpers

# Merge labels of data points with identical smile - before standardisation
smiles_df = helpers.read_df("../data/tox21/full/standardised/smiles.csv")
activity_df = helpers.read_df("../data/tox21/full/standardised/activity.csv")

merged_smiles_df, merged_activity_df, smiles_dict, mapping_dict = merge.merge_labels(smiles_df, activity_df)

helpers.write_df(merged_smiles_df, "../data/tox21/full/merged/smiles.csv")
helpers.write_df(merged_activity_df, "../data/tox21/full/merged/activity.csv")
helpers.save_pickle(smiles_dict, "../data/tox21/full/merged/smiles_dict.pickle")
helpers.save_pickle(mapping_dict, "../data/tox21/full/merged/mapping_dict.pickle")

# Rename ids in each set
train_ids_df = helpers.read_df("../data/tox21/full/train_ids.csv")
train_ids_df = merge.rename_ids(train_ids_df, mapping_dict)
helpers.write_df(train_ids_df, "../data/tox21/full/merged/ids/train_ids.csv")

leader_ids_df = helpers.read_df("../data/tox21/full/leader_ids.csv")
leader_ids_df = merge.rename_ids(leader_ids_df, mapping_dict)
helpers.write_df(leader_ids_df, "../data/tox21/full/merged/ids/leader_ids.csv")

eval_ids_df = helpers.read_df("../data/tox21/full/eval_ids.csv")
eval_ids_df = merge.rename_ids(eval_ids_df, mapping_dict)
helpers.write_df(eval_ids_df, "../data/tox21/full/merged/ids/eval_ids.csv")

# Remove repeated compounds (in eval set) from the train and leader sets
train_ids_df = helpers.read_df("../data/tox21/full/merged/ids/train_ids.csv")
eval_ids_df = helpers.read_df("../data/tox21/full/merged/ids/eval_ids.csv")
train_ids_df = merge.remove_duplicates(train_ids_df, eval_ids_df)
helpers.write_df(train_ids_df, "../data/tox21/full/merged/ids/train_filtered_ids.csv")

leader_ids_df = helpers.read_df("../data/tox21/full/merged/ids/leader_ids.csv")
eval_ids_df = helpers.read_df("../data/tox21/full/merged/ids/eval_ids.csv")
leader_ids_df = merge.remove_duplicates(leader_ids_df, eval_ids_df)
helpers.write_df(leader_ids_df, "../data/tox21/full/merged/ids/leader_filtered_ids.csv")

# Concatenate train and leader datasets
train_leader_ids_df = pd.concat([train_ids_df, leader_ids_df])
helpers.write_df(train_leader_ids_df, "../data/tox21/full/merged/ids/train_leader_ids.csv")
