import pandas as pd

def clean_cddd_df(cddd_df, remove_columns):
    cddd_df = cddd_df.drop(columns=cddd_df.columns[remove_columns])
    cddd_df = cddd_df.dropna()
    return cddd_df
