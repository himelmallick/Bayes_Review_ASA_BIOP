import src.data.smiles as smiles
import src.utils.helpers as helpers

# Standardise fragments and choose one, when multiple fragments present
smiles_df = helpers.read_df("../data/tox21/full/smiles.csv")
stand_smiles_df = smiles.standardize_compounds(smiles_df, remove_charge=True)
helpers.write_df(stand_smiles_df, "../data/tox21/full/standardised/smiles.csv")

# Update activity to exclude invalid compounds
activity_df = helpers.read_df("../data/tox21/full/activity.csv")
activity_df = activity_df.loc[stand_smiles_df.index]
helpers.write_df(activity_df, "../data/tox21/full/standardised/activity.csv")
