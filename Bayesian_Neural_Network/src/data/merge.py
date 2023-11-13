import numpy as np
import pandas as pd
import sys
sys.path.append(".")

from utils.constants import SMILE, UNKNOWN, ID

# Methods to merge labels for data points with identical smile

def merge_labels(smiles_df, activity_df):
    """ Merges labels of data points identical smiles
        Sets to np.nan if conflict occurs

    Parameters
    ----------
    smiles_df: pandas.DataFrame
        Smiles indexed by compound id
    activity_df: pandas.DataFrame
        Activity labels indexed by compound id

    Returns
    -------
    merged_smiles_df: pandas.DataFrame
    merged_activity_df: pandas.DataFrame
    smiles_dict: dictionary
        Mapping from smiles to lists of compound ids
    """
    # activity_df.replace(UNKNOWN, np.nan, inplace=True)
    # activity_df = activity_df.apply(pd.to_numeric)
    smiles_dict = {}
    mapping_dict = {}
    for id, row in smiles_df.iterrows():
        smile = row[SMILE]
        add_smile(smiles_dict, smile, id)

    merged_activity_df = pd.DataFrame().reindex_like(activity_df)
    num_conflicts = 0 # The number of labelling conflicts

    for smile, smile_ids in smiles_dict.items():
        # The first id in the list is selected
        new_id = smile_ids[0]
        for smile_id in smile_ids:
            mapping_dict[smile_id] = new_id
        merged_labels, add_num_conflicts = merge_all_labels(activity_df, smile_ids)
        merged_activity_df.loc[new_id] = merged_labels
        num_conflicts += add_num_conflicts

    merged_activity_df.dropna(axis="index", how="all", inplace=True)
    merged_smiles_df = smiles_df.copy()
    merged_smiles_df = merged_smiles_df.loc[merged_activity_df.index]

    print("Number of labels: " + str(np.sum(merged_activity_df.count())))
    print("Number of conflicts: " + str(num_conflicts))

    return merged_smiles_df, merged_activity_df, smiles_dict, mapping_dict

def add_smile(smiles_dict, smile, id):
    if smile not in smiles_dict:
        smiles_dict[smile] = []
    smiles_dict[smile].append(id)

def merge_all_labels(activity_df, ids):
    current_labels = activity_df.loc[ids[0]].copy()
    num_conflicts = 0
    if len(ids) > 1:
        skip_assays = set()
        for id in ids[1:]:
            new_labels = activity_df.loc[id].copy()
            current_labels, skip_assays, add_num_conflicts = merge_two_labels(current_labels, new_labels, skip_assays)
            num_conflicts += add_num_conflicts
    return current_labels, num_conflicts

def merge_two_labels(current_labels, new_labels, skip_assays):
    merged_labels = current_labels.copy()
    num_conflicts = 0
    for assay, current_label in current_labels.items():
        if assay in skip_assays:
            continue

        new_label = new_labels[assay]
        # Check if the new label for the assay is unset
        if np.isnan(new_label):
            continue

        if np.isnan(current_label): # Label for the assay is not set yet
            merged_labels[assay] = new_label

        elif current_label != new_label: # Label for the assay is already set and the labels don't agree
            merged_labels[assay] = np.nan
            skip_assays.add(assay)
            num_conflicts += 1
    return merged_labels, skip_assays, num_conflicts

def rename_ids(ids_df, mapping_dict):
    new_ids_df = ids_df.copy()

    for id, _ in ids_df.iterrows():
        if id not in mapping_dict:
            new_ids_df.drop(index=id, inplace=True)

    new_ids_df = new_ids_df.rename(index=mapping_dict)

    new_ids_df.reset_index(inplace=True)
    new_ids_df.drop_duplicates(inplace=True)
    new_ids_df.set_index(ID, inplace=True)
    return new_ids_df

def remove_duplicates(ids1_df, ids2_df):
    # Remove those ids from ids1_df that are also in ids2_df
    ids1 = set(ids1_df.index)
    ids2 = set(ids2_df.index)
    ids1_df = ids1_df.drop(ids1.intersection(ids2))
    return ids1_df
