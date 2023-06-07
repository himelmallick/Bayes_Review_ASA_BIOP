from molvs import Standardizer
import pandas as pd
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from src.utils.constants import ID, ORGANIC_ATOMIC_NUMS, CARBON_ATOMIC_NUM, SMILE

# Methods to standardize all fragments in a compound and pick a single fragment

def standardize_compounds(smiles_df, remove_charge=False, keep_all=False):
    """ Standardize all compounds in a dataframe and return a new (potentially
        reduced) dataframe
    """
    standardizer = Standardizer()
    salt_remover = SaltRemover()

    stand_smiles_df = pd.DataFrame().reindex_like(smiles_df)

    i = 0
    for id, row in smiles_df.iterrows():
        print(i)
        i += 1
        smile = row[SMILE]
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                continue
            stand_mol = standardize_compound(mol, standardizer, salt_remover, remove_charge, keep_all)
            if stand_mol is not None:
                smile = Chem.MolToSmiles(stand_mol, isomericSmiles=False, canonical=True)
                stand_smiles_df.loc[id] = smile
            if stand_mol is None and keep_all:
                smile = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
                stand_smiles_df.loc[id] = smile
        except Exception as e:
            print(e)
    stand_smiles_df = stand_smiles_df.dropna(axis="index", how="all")
    return stand_smiles_df

def standardize_compound(mol, standardizer, salt_remover, remove_charge, keep_all):
    """ Standardize the given compound to provide a single fragment using the
        following steps:
        1. Remove salts using SaltRemover
        2. Discard all inorganic fragments
        3. If more than one organic fragment, discard data point
        4. Optionally remove charges
    """
    mol = standardize_mol(mol, standardizer)
    mol = salt_remover.StripMol(mol)
    if mol.GetNumAtoms()==0:
        return None
    fragments = Chem.GetMolFrags(mol, asMols=True)
    selected_fragment = None

    # for fragment in fragments:
    #     if is_organic(fragment):
    #         if selected_fragment is None:
    #             selected_fragment = fragment
    #         else: # Organic fragment was already found i.e. there are multiple organic fragments
    #             selected_fragment = None
    #             break

    # if selected_fragment is None:
    #     return None

    organic_fragments = []
    for fragment in fragments:
        if is_organic(fragment):
            organic_fragments.append(fragment)

    if len(organic_fragments) == 0:
        if not keep_all:
            return None
        selected_fragment = select_largest_fragment(fragments)

    if len(organic_fragments) == 1:
        selected_fragment = organic_fragments[0]

    if len(organic_fragments) > 1:
        if not keep_all:
            return None
        selected_fragment = select_largest_fragment(organic_fragments)

    if remove_charge:
        selected_fragment = remove_charge_mol(selected_fragment, standardizer)

    return selected_fragment

def select_largest_fragment(fragments):
    largest_fragment = None
    largest_size = -1

    for fragment in fragments:
        size = fragment.GetNumAtoms()
        if size > largest_size:
            largest_fragment = fragment
            largest_size = size
    
    return largest_fragment

def standardize_mol(mol, st):
    """ Equivalent to MolVS super_parent method but without charge_parent and
        isotope_parent methods applied

        Parameters
        mol: rdkit.Chem.rdchem.Mol
        st: molvs.standardize.Standardizer
    """
    mol = st.standardize(mol)
    mol = st.stereo_parent(mol, skip_standardize=True)
    mol = st.tautomer_parent(mol, skip_standardize=True)
    mol = st.standardize(mol)
    return mol

def remove_charge_mol(mol, st):
    # Should remove_charge be applied?
    return st.charge_parent(mol, skip_standardize=False)

def is_organic(fragment, organic_atomic_nums=ORGANIC_ATOMIC_NUMS, carbon_atomic_num=CARBON_ATOMIC_NUM):
    """ Return true if fragment contains at least one carbon atom
        and all atoms are in the {H, C, N, O, S, P, F, Cl, Br, I} set.

        Parameters
        fragment: rdkit.Chem.rdchem.Mol
    """
    contains_carbon = False
    in_organic_set = True
    for atom in fragment.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num not in organic_atomic_nums:
            in_organic_set = False
            break
        if atomic_num == carbon_atomic_num:
            contains_carbon = True
    return contains_carbon and in_organic_set
