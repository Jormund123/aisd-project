"""
Curated seed molecules for LLM prompts.
Selected from AqSolDB: chemically diverse, short SMILES, full LogS range coverage.
LogS values are experimental (from AqSolDB column Y).
"""

SEED_MOLECULES = [
    # High solubility (LogS > 1.0)
    {"name": "acetamide",    "smiles": "CC(N)=O",             "logs": 1.581},
    {"name": "methanol",     "smiles": "CO",                  "logs": 1.494},
    {"name": "ethanol",      "smiles": "CCO",                 "logs": 1.234},
    {"name": "acetone",      "smiles": "CC(C)=O",             "logs": 1.236},

    # Moderate solubility (-1.5 to -0.0)
    {"name": "phenol",       "smiles": "Oc1ccccc1",           "logs": -0.040},
    {"name": "aniline",      "smiles": "Nc1ccccc1",           "logs": -0.425},
    {"name": "caffeine",     "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "logs": -0.910},

    # Low solubility (-2.5 to -1.5)
    {"name": "toluene",      "smiles": "Cc1ccccc1",           "logs": -2.206},

    # Poor solubility (-4.0 to -2.5)
    {"name": "cyclohexane",  "smiles": "C1CCCCC1",            "logs": -3.100},
    {"name": "hexane",       "smiles": "CCCCCC",              "logs": -3.944},
    {"name": "naphthalene",  "smiles": "c1ccc2ccccc2c1",      "logs": -4.311},

    # Very poor solubility (< -5.0)
    {"name": "1-decene",     "smiles": "C=CCCCCCCCC",         "logs": -5.510},
    {"name": "stilbene",     "smiles": "C(=C/c1ccccc1)/c1ccccc1", "logs": -5.791},
]
