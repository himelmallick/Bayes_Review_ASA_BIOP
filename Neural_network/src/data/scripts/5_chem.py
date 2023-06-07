from src.utils.constants import CHEM_FTRS
import src.data.chem as chem
import src.utils.helpers as helpers

# Obtain chemical features
# Make Ipc feature to log(Ipc) to reduce its scale
smiles_df = helpers.read_df("../data/tox21/full/merged/smiles.csv")
ftrs_df = chem.get_chem_ftrs(smiles_df, CHEM_FTRS)
ftrs_df = chem.make_log_feature(ftrs_df, ftr_name="Ipc")
helpers.write_df(ftrs_df, "../data/tox21/full/chem/chem.csv")
