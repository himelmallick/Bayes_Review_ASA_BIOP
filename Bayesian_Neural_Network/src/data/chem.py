# from molvs import Standardizer
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from src.utils.constants import ID, SMILE

def get_chem_ftrs(smiles_df, ftrs_list):#, use_stand=False):
    ftrs_df = pd.DataFrame(columns = [ID] + ftrs_list)
    ftrs_df.set_index(ID, inplace=True)
    ftrs_calc = MolecularDescriptorCalculator(ftrs_list)
    # stand = Standardizer()

    for id, row in smiles_df.iterrows():
        # print(i)
        # id = row[ID]
        smile = row[SMILE]

        # 1. Convert smile to RDKit mol object.
        try:
            mol = Chem.MolFromSmiles(smile)
        except:
            print("Failed to convert.")
            continue

        # # 2. Standardise the structure.
        # if use_stand:
        #     try:
        #         mol = stand.standardize(mol)
        #     except:
        #         print("Failed to standardize.")
        #         continue

        # 3. Calculate the descriptors.
        try:
            ftrs = list(ftrs_calc.CalcDescriptors(mol))
            ftrs_df.loc[id] =  ftrs
        except Exception as e:
            print(e)
            print("Failed to calculate descriptors.")
            continue

    return ftrs_df

def make_log_feature(ftrs_df, ftr_name, eps=0.001):
    ftrs_df["Log_" + ftr_name] = np.log(ftrs_df.pop(ftr_name) + eps)
    return ftrs_df
