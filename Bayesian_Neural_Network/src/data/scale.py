import numpy as np
import pandas as pd
from sklearn import preprocessing

def scale_cts_features(ftrs_df, feature_range=None, scaler=None):
    if scaler==None:
        scaler = preprocessing.StandardScaler()
        scaler.fit(ftrs_df)
    ftrs_np = scaler.transform(ftrs_df)

    if feature_range is not None:
        ftrs_np = np.clip(ftrs_np, feature_range[0], feature_range[1])

    ftrs_df = pd.DataFrame(ftrs_np, columns=ftrs_df.columns, index=ftrs_df.index)

    return ftrs_df, scaler

def scale_cts_disc_ftrs(ftrs_df, feature_range, scalers=None, ftrs_lists=None):
    # Retrieve lists of feature names and scalers
    if scalers is None:
        cts_ftrs_list, disc_ftrs_list = get_ftrs_lists(ftrs_df)
        # Continuous scaler
        cts_scaler = preprocessing.StandardScaler()
        cts_scaler.fit(ftrs_df[cts_ftrs_list])
        # Discrete scaler
        disc_scaler = preprocessing.MinMaxScaler(feature_range)
        disc_scaler.fit(ftrs_df[disc_ftrs_list])
    else:
        cts_ftrs_list = ftrs_lists[0]
        disc_ftrs_list = ftrs_lists[1]
        cts_scaler = scalers[0]
        disc_scaler = scalers[1]

    scaled_ftrs_df = ftrs_df.copy()

    cts_ftrs = cts_scaler.transform(ftrs_df[cts_ftrs_list])
    cts_ftrs = np.clip(cts_ftrs, feature_range[0], feature_range[1])
    scaled_ftrs_df[cts_ftrs_list] = cts_ftrs

    disc_ftrs = disc_scaler.transform(ftrs_df[disc_ftrs_list])
    scaled_ftrs_df[disc_ftrs_list] = disc_ftrs

    return scaled_ftrs_df, (cts_scaler, disc_scaler), (cts_ftrs_list, disc_ftrs_list)

def get_ftrs_lists(ftrs_df):
    cts_ftrs = []
    disc_ftrs = []
    is_int = lambda num: num%1==0

    for ftr in ftrs_df:
        vals = ftrs_df[ftr]
        if all(is_int(val) for val in vals):
            disc_ftrs.append(ftr)
        else:
            cts_ftrs.append(ftr)

    return cts_ftrs, disc_ftrs
