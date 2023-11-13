# Chemical descriptors used
import numpy as np

CHEM_FTRS_ALL = ["Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v",
            "Chi3n", "Chi3v", "Chi4n", "Chi4v", "EState_VSA1", "EState_VSA10",
            "EState_VSA11", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
            "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "FractionCSP3",
            "HallKierAlpha", "HeavyAtomCount", "Ipc", "Kappa1", "Kappa2", "Kappa3",
            "LabuteASA", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount", "NumAliphaticCarbocycles",
            "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles",
            "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
            "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "PEOE_VSA1",
            "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2",
            "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9",
            "RingCount", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6",
            "SMR_VSA7", "SMR_VSA8", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
            "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
            "SlogP_VSA9", "TPSA", "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4",
            "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9"]
# Features SMR_VSA8 and SlogP_VSA9 have variance zero on train set so removed
CHEM_FTRS = ["Chi0", "Chi0n", "Chi0v", "Chi1", "Chi1n", "Chi1v", "Chi2n", "Chi2v",
             "Chi3n", "Chi3v", "Chi4n", "Chi4v", "EState_VSA1", "EState_VSA10",
             "EState_VSA11", "EState_VSA2", "EState_VSA3", "EState_VSA4", "EState_VSA5",
             "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9", "FractionCSP3",
             "HallKierAlpha", "HeavyAtomCount", "Ipc", "Kappa1", "Kappa2", "Kappa3",
             "LabuteASA", "MolLogP", "MolMR", "MolWt", "NHOHCount", "NOCount", "NumAliphaticCarbocycles",
             "NumAliphaticHeterocycles", "NumAliphaticRings", "NumAromaticCarbocycles", "NumAromaticHeterocycles",
             "NumAromaticRings", "NumHAcceptors", "NumHDonors", "NumHeteroatoms", "NumRotatableBonds",
             "NumSaturatedCarbocycles", "NumSaturatedHeterocycles", "NumSaturatedRings", "PEOE_VSA1",
             "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14", "PEOE_VSA2",
             "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9",
             "RingCount", "SMR_VSA1", "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6",
             "SMR_VSA7", "SMR_VSA9", "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12",
             "SlogP_VSA2", "SlogP_VSA3", "SlogP_VSA4", "SlogP_VSA5", "SlogP_VSA6", "SlogP_VSA7", "SlogP_VSA8",
             "TPSA", "VSA_EState1", "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4",
             "VSA_EState5", "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9"]

# The list of all 12 assays (7 NR, 5 SR)
ASSAYS = ["NR-AhR", "NR-AR", "NR-AR-LBD", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
          "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

ASSAYS_LOWER = [assay.lower() for assay in ASSAYS]
NUM_ASSAYS = 12

# Data frame column names
ID = "id" # Identification of compounds
ID_SDF = "Compound Batch ID"
SMILE = "smile"
ACTIVITY = "activity"

# Assay activities
ACTIVE = 1
INACTIVE = 0
UNKNOWN = np.nan
EMPTY_ACT_LIST = [UNKNOWN] * NUM_ASSAYS

# Organic atomic numbers to determine if a fragment is organic
ORGANIC_ATOMIC_NUMS = set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53])
CARBON_ATOMIC_NUM = 6
