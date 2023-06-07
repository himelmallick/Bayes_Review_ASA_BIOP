import src.data.chem as chem
import src.data.smiles as smiles
import src.utils.helpers as helpers
import src.utils.helpers as helpers
from src.utils.constants import CHEM_FTRS
import src.data.assay as assay
import src.data.scale as scale

# # Standardise test compounds and keep all
# smiles_df = helpers.read_df("/home/kxkr044/bnns_for_tox21/data/tox21/interim/1_separate/eval/smiles.csv")
# stand_smiles_df = smiles.standardize_compounds(smiles_df, remove_charge=False, keep_all=True)
# helpers.write_df(stand_smiles_df, "/home/kxkr044/bnns_for_tox21/data/tox21/interim/3_orig_test/standardised/smiles.csv")

# # Update activity to exclude invalid compounds
# activity_df = helpers.read_df("/home/kxkr044/bnns_for_tox21/data/tox21/interim/1_separate/eval/activity.csv")
# activity_df = activity_df.loc[stand_smiles_df.index]
# helpers.write_df(activity_df, "/home/kxkr044/bnns_for_tox21/data/tox21/interim/3_orig_test/standardised/activity.csv")

# # Obtain chemical features
# # Make Ipc feature to log(Ipc) to reduce its scale
# smiles_df = helpers.read_df("/home/kxkr044/bnns_for_tox21/data/tox21/interim/3_orig_test/standardised/smiles.csv")
# ftrs_df = chem.get_chem_ftrs(smiles_df, CHEM_FTRS)
# ftrs_df = chem.make_log_feature(ftrs_df, ftr_name="Ipc")
# helpers.write_df(ftrs_df, "/home/kxkr044/bnns_for_tox21/data/tox21/interim/3_orig_test/standardised/chem.csv")

# Select chem features for the mmp assay
activity_df = helpers.read_df("/home/kxkr044/bnns_for_tox21/data/tox21/interim/3_orig_test/standardised/activity.csv")
chem_df = helpers.read_df("/home/kxkr044/bnns_for_tox21/data/tox21/interim/3_orig_test/standardised/chem.csv")
assay_id = "sr-mmp"
feature_range = (-5, 5)

# Eval/Test
smiles_df = helpers.read_df("/home/kxkr044/bnns_for_tox21/data/tox21/interim/3_orig_test/standardised/smiles.csv")
activity_df = assay.select_activity(activity_df, smiles_df, assay_name=assay_id)
activity_df, chem_df = assay.select_features(activity_df, chem_df)

helpers.write_df(activity_df, "/home/kxkr044/bnns_for_tox21/data/tox21/processed/chem/sr/mmp/test_orig_activity.csv")
helpers.write_df(chem_df, "/home/kxkr044/bnns_for_tox21/data/tox21/processed/chem/sr/mmp/test_orig_chem.csv")

# Scale
cts_scaler = helpers.read_pickle("/home/kxkr044/bnns_for_tox21/data/tox21/processed/chem/sr/mmp/cts_scaler.pickle")
disc_scaler = helpers.read_pickle("/home/kxkr044/bnns_for_tox21/data/tox21/processed/chem/sr/mmp/disc_scaler.pickle")
scalers = (cts_scaler, disc_scaler)

cts_ftrs_list = helpers.read_list("/home/kxkr044/bnns_for_tox21/data/tox21/processed/chem/sr/mmp/cts_ftrs_list.csv")
disc_ftrs_list = helpers.read_list("/home/kxkr044/bnns_for_tox21/data/tox21/processed/chem/sr/mmp/disc_ftrs_list.csv")
ftrs_lists = (cts_ftrs_list, disc_ftrs_list)

scaled_chem_df, _, _ = scale.scale_cts_disc_ftrs(chem_df, feature_range, scalers=scalers, ftrs_lists=ftrs_lists)
helpers.write_df(scaled_chem_df, "/home/kxkr044/bnns_for_tox21/data/tox21/processed/chem/sr/mmp/scaled_test_orig_chem.csv")