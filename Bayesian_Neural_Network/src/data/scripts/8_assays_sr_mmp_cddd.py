import src.data.assay as assay
import src.data.scale as scale
import src.utils.helpers as helpers

# Select cddd features for the mmp assay
activity_df = helpers.read_df("data/tox21/interim/2_full/3_merged/activity.csv")
cddd_df = helpers.read_df("data/tox21/interim/2_full/4_cddd/cddd.csv")
assay_id = "sr-mmp"
feature_range = (-5, 5)

# Train + leader
train_leader_ids_df = helpers.read_df("data/tox21/interim/2_full/3_merged/ids/train_leader_ids.csv")
train_leader_activity_df = assay.select_activity(activity_df, train_leader_ids_df, assay_name=assay_id)
train_leader_activity_df, train_leader_cddd_df = assay.select_features(train_leader_activity_df, cddd_df)

helpers.write_df(train_leader_activity_df, "data/tox21/processed/cddd/sr/mmp/train_leader_activity.csv")
helpers.write_df(train_leader_cddd_df, "data/tox21/processed/cddd/sr/mmp/train_leader_cddd.csv")

# Split train + leader to train + val
train_activity_df, val_activity_df, train_cddd_df, val_cddd_df = assay.get_train_test_split(train_leader_activity_df, train_leader_cddd_df, test_size=0.1, stratify_col=assay_id)

helpers.write_df(train_activity_df, "data/tox21/processed/cddd/sr/mmp/train_activity.csv")
helpers.write_df(train_cddd_df, "data/tox21/processed/cddd/sr/mmp/train_cddd.csv")
helpers.write_df(val_activity_df, "data/tox21/processed/cddd/sr/mmp/val_activity.csv")
helpers.write_df(val_cddd_df, "data/tox21/processed/cddd/sr/mmp/val_cddd.csv")

# Eval/Test
test_ids_df = helpers.read_df("data/tox21/interim/2_full/3_merged/ids/eval_ids.csv")
test_activity_df = assay.select_activity(activity_df, test_ids_df, assay_name=assay_id)
test_activity_df, test_cddd_df = assay.select_features(test_activity_df, cddd_df)

helpers.write_df(test_activity_df, "data/tox21/processed/cddd/sr/mmp/test_activity.csv")
helpers.write_df(test_cddd_df, "data/tox21/processed/cddd/sr/mmp/test_cddd.csv")

# Scale
# Train
scaled_train_cddd_df, scaler = scale.scale_cts_features(train_cddd_df, feature_range, scaler=None)
helpers.write_df(scaled_train_cddd_df, "data/tox21/processed/cddd/sr/mmp/scaled_train_cddd.csv")

# Val
scaled_val_cddd_df, _ = scale.scale_cts_features(val_cddd_df, feature_range, scaler)
helpers.write_df(scaled_val_cddd_df, "data/tox21/processed/cddd/sr/mmp/scaled_val_cddd.csv")

# Test
scaled_test_cddd_df, _ = scale.scale_cts_features(test_cddd_df, feature_range, scaler)
helpers.write_df(scaled_test_cddd_df, "data/tox21/processed/cddd/sr/mmp/scaled_test_cddd.csv")

# Save scaler
helpers.write_pickle(scaler, "data/tox21/processed/cddd/sr/mmp/scaler.pickle")