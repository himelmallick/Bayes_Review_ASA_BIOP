import numpy as np
import pandas as pd
from sklearn import model_selection
import sys
sys.path.append(".")

def select_activity(activity_df, ids_df, assay_name):
    assay_activity_df = activity_df.loc[ids_df.index]
    assay_activity_df = assay_activity_df[[assay_name]]
    assay_activity_df = assay_activity_df.dropna()
    return assay_activity_df

def select_features(assay_activity_df, ftrs_df):
    # Select compounds with available activity and chemical features for the given assay
    assay_compounds = ftrs_df.index.intersection(assay_activity_df.index)
    assay_ftrs_df = ftrs_df.loc[assay_compounds]
    assay_activity_df = assay_activity_df.loc[assay_compounds]
    return assay_activity_df, assay_ftrs_df

def merge_features(assay_activity_df, ftrs1_df, ftrs2_df):
    assay_activity_df, ftrs1_df = select_features(assay_activity_df, ftrs1_df)
    assay_activity_df, ftrs2_df = select_features(assay_activity_df, ftrs2_df)
    assay_activity_df, ftrs1_df = select_features(assay_activity_df, ftrs1_df)
    ftrs_df = ftrs1_df.join(ftrs2_df)
    return assay_activity_df, ftrs_df

def get_train_test_split(activity_df, ftrs_df, test_size, stratify_col):
    train_activity_df, test_activity_df = model_selection.train_test_split(activity_df, test_size=test_size, stratify=activity_df[stratify_col])
    train_ftrs_df = ftrs_df.loc[train_activity_df.index]
    test_ftrs_df = ftrs_df.loc[test_activity_df.index]
    return train_activity_df, test_activity_df, train_ftrs_df, test_ftrs_df
