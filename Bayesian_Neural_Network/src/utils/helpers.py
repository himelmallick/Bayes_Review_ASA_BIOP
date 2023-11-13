import json
import pickle
import os
import pandas as pd

from src.utils.constants import ID

def listdir_fullpath(dir):
    """Returns a list of files"""
    files_list = [os.path.join(dir, file) for file in os.listdir(dir)]
    return files_list

def write_df(df, df_file):
    df.to_csv(df_file, sep=",", header=True, index=True)

def read_df(df_file, index=ID):
    df = pd.read_csv(df_file, sep=",", header=0)
    if index is not None:
        df.set_index(index, inplace=True)
    return df

def write_pickle(obj, obj_file):
    with open(obj_file, "wb") as file:
        pickle.dump(obj, file, protocol=4)

def read_pickle(obj_file):
    with open(obj_file, "rb") as file:
        obj = pickle.load(file)
    return obj

def write_list(list, list_file):
    with open(list_file, "w") as file:
        for item in list:
            file.write("%s\n" % item)

def read_list(list_file):
    list = []
    with open(list_file, "r") as file:
        for row in file:
            list.append(row.strip())
    return list

def save_config(config, config_file):
    with open(config_file, "w") as file:
        json.dump(config, file, indent=4)
        file.truncate()

def read_config(config_file):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config