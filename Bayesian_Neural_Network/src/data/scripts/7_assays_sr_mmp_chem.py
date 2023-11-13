import src.data.assay as assay
import src.data.scale as scale
import src.utils.helpers as helpers

# Select chem features for the mmp assay
activity_df = helpers.read_df("data/tox21/interim/2_full/3_merged/activity.csv")
chem_df = helpers.read_df("data/tox21/interim/2_full/4_chem/chem.csv")
assay_id = "sr-mmp"
feature_range = (-5, 5)

# Train + leader
train_leader_ids_df = helpers.read_df("data/tox21/interim/2_full/3_merged/ids/train_leader_ids.csv")
train_leader_activity_df = assay.select_activity(activity_df, train_leader_ids_df, assay_name=assay_id)
train_leader_activity_df, train_leader_chem_df = assay.select_features(train_leader_activity_df, chem_df)

helpers.write_df(train_leader_activity_df, "data/tox21/processed/cddd/sr/mmp/train_leader_activity.csv")
helpers.write_df(train_leader_chem_df, "data/tox21/processed/cddd/sr/mmp/train_leader_chem.csv")

# Split train + leader to train + val
train_activity_df, val_activity_df, train_chem_df, val_chem_df = assay.get_train_test_split(train_leader_activity_df, train_leader_chem_df, test_size=0.1, stratify_col=assay_id)

helpers.write_df(train_activity_df, "data/tox21/processed/cddd/sr/mmp/train_activity.csv")
helpers.write_df(train_chem_df, "data/tox21/processed/cddd/sr/mmp/train_chem.csv")
helpers.write_df(val_activity_df, "data/tox21/processed/cddd/sr/mmp/val_activity.csv")
helpers.write_df(val_chem_df, "data/tox21/processed/cddd/sr/mmp/val_chem.csv")

# Eval/Test
test_ids_df = helpers.read_df("data/tox21/interim/2_full/3_merged/ids/eval_ids.csv")
test_activity_df = assay.select_activity(activity_df, test_ids_df, assay_name=assay_id)
test_activity_df, test_chem_df = assay.select_features(test_activity_df, chem_df)

helpers.write_df(test_activity_df, "data/tox21/processed/cddd/sr/mmp/test_activity.csv")
helpers.write_df(test_chem_df, "data/tox21/processed/cddd/sr/mmp/test_chem.csv")

# Scale
# Train
scaled_train_chem_df, scalers, ftrs_lists = scale.scale_cts_disc_ftrs(train_chem_df, feature_range, scalers=None, ftrs_lists=None)
helpers.write_df(scaled_train_chem_df, "data/tox21/processed/cddd/sr/mmp/scaled_train_chem.csv")

# Val
scaled_val_chem_df, _, _ = scale.scale_cts_disc_ftrs(val_chem_df, feature_range, scalers=scalers, ftrs_lists=ftrs_lists)
helpers.write_df(scaled_val_chem_df, "data/tox21/processed/cddd/sr/mmp/scaled_val_chem.csv")

# Test
scaled_test_chem_df, _, _ = scale.scale_cts_disc_ftrs(test_chem_df, feature_range, scalers=scalers, ftrs_lists=ftrs_lists)
helpers.write_df(scaled_test_chem_df, "data/tox21/processed/cddd/sr/mmp/scaled_test_chem.csv")

# Save scalers and lists of features
helpers.save_pickle(scalers[0], "data/tox21/processed/cddd/sr/mmp/cts_scaler.pickle")
helpers.save_pickle(scalers[1], "data/tox21/processed/cddd/sr/mmp/disc_scaler.pickle")
helpers.write_list(ftrs_lists[0], "data/tox21/processed/cddd/sr/mmp/cts_ftrs_list.csv")
helpers.write_list(ftrs_lists[1], "data/tox21/processed/cddd/sr/mmp/disc_ftrs_list.csv")
