import src.data.cddd as cddd
import src.utils.helpers as helpers

# conda activate cddd
# cddd --input data/tox21/interim/full/merged/smiles.csv --output data/tox21/interim/full/cddd/cddd_orig2.csv --smiles_header smile

# Remove numerical index and two smiles columns in the cddd features df
remove_columns_smiles = [0] # Keep smiles in the df
remove_columns = [0, 1, 2] # Remove empty column and smiles

# Train data
cddd_orig_df = helpers.read_df("data/tox21/interim/2_full/4_cddd/cddd_orig.csv")

cddd_smiles_df = cddd.clean_cddd_df(cddd_orig_df, remove_columns=[0])
helpers.write_df(cddd_smiles_df, "data/tox21/interim/2_full/4_cddd/cddd_smiles.csv")

cddd_df = cddd.clean_cddd_df(cddd_orig_df, remove_columns=[0, 1, 2])
helpers.write_df(cddd_df, "data/tox21/interim/2_full/4_cddd/cddd.csv")