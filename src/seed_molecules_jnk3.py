"""
Curated seed molecules for JNK3 prompts.
Hand-picked from data/jnk3_scored_pool.csv (a 3000-molecule random sample of ZINC250k,
see build_jnk3_seed_pool.py) to span the oracle's score range while staying short and
readable. Unlike the ESOL seeds these have no common names (anonymous ZINC catalog
entries), so they're numbered instead. jnk3 = TDC Oracle('JNK3') probability [0, 1].
"""

SEED_MOLECULES = [
    {"name": "zinc_01", "smiles": "COc1ccc(C=C(C#N)C#N)cc1O", "jnk3": 0.310},
    {"name": "zinc_02", "smiles": "NC(=O)c1ccc(Nc2ncnc3scc(-c4ccccc4)c23)cc1", "jnk3": 0.220},
    {"name": "zinc_03", "smiles": "COC(=O)c1cc(C)ccc1NC(=O)c1cccc(Nc2cnn(C)c2)c1", "jnk3": 0.180},
    {"name": "zinc_04", "smiles": "Nc1c(NCc2ccccc2)ncnc1Nc1ccccc1F", "jnk3": 0.100},
    {"name": "zinc_05", "smiles": "O=[N+]([O-])c1ccc(-c2ccc(CO)cc2)cc1", "jnk3": 0.100},
    {"name": "zinc_06", "smiles": "N#Cc1c(Cn2cc(I)cn2)cn2ccccc12", "jnk3": 0.090},
    {"name": "zinc_07", "smiles": "O=c1ccoc2c(Cl)cc(Cl)cc12", "jnk3": 0.060},
    {"name": "zinc_08", "smiles": "COc1ccc(Nc2cc(C)ccc2N)cc1C", "jnk3": 0.050},
    {"name": "zinc_09", "smiles": "CC(=O)Nc1cc(C)c(N)cn1", "jnk3": 0.040},
    {"name": "zinc_10", "smiles": "O=[S@@]1CCCNc2ccccc21", "jnk3": 0.020},
    {"name": "zinc_11", "smiles": "N#Cc1c[nH]c2ccc(Cl)cc12", "jnk3": 0.020},
    {"name": "zinc_12", "smiles": "Cc1ncccc1NC(=O)OCCO", "jnk3": 0.010},
    {"name": "zinc_13", "smiles": "Cn1nc(Br)cc1N", "jnk3": 0.000},
    {"name": "zinc_14", "smiles": "CSCCc1ccc(N)cc1", "jnk3": 0.000},
]
