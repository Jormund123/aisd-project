"""
Seed molecules: the few-shot examples shown to the LLM at the start of every run, for
both properties. Names + SMILES only -- scores are always computed fresh via the
property registry (common.py), never trusted as a stored number, so there's one
source of truth (the oracle) instead of two that could drift apart.

ESOL seeds: picked from AqSolDB, span the full LogS range, short/readable SMILES.
JNK3 seeds: picked from a 3000-molecule ZINC250k sample scored with the JNK3 oracle
(see docs/jnk3_implementation_plan.md) -- span its score range (0.0 to 0.31); most
random molecules score near 0, so these are the diverse, discriminating ones.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import PROPERTIES  # noqa: E402

_RAW = {
    "esol": [
        {"name": "acetamide",    "smiles": "CC(N)=O"},
        {"name": "methanol",     "smiles": "CO"},
        {"name": "ethanol",      "smiles": "CCO"},
        {"name": "acetone",      "smiles": "CC(C)=O"},
        {"name": "phenol",       "smiles": "Oc1ccccc1"},
        {"name": "aniline",      "smiles": "Nc1ccccc1"},
        {"name": "caffeine",     "smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O"},
        {"name": "toluene",      "smiles": "Cc1ccccc1"},
        {"name": "cyclohexane",  "smiles": "C1CCCCC1"},
        {"name": "hexane",       "smiles": "CCCCCC"},
        {"name": "naphthalene",  "smiles": "c1ccc2ccccc2c1"},
        {"name": "1-decene",     "smiles": "C=CCCCCCCCC"},
        {"name": "stilbene",     "smiles": "C(=C/c1ccccc1)/c1ccccc1"},
    ],
    "jnk3": [
        {"name": "zinc_01", "smiles": "COc1ccc(C=C(C#N)C#N)cc1O"},
        {"name": "zinc_02", "smiles": "NC(=O)c1ccc(Nc2ncnc3scc(-c4ccccc4)c23)cc1"},
        {"name": "zinc_03", "smiles": "COC(=O)c1cc(C)ccc1NC(=O)c1cccc(Nc2cnn(C)c2)c1"},
        {"name": "zinc_04", "smiles": "Nc1c(NCc2ccccc2)ncnc1Nc1ccccc1F"},
        {"name": "zinc_05", "smiles": "O=[N+]([O-])c1ccc(-c2ccc(CO)cc2)cc1"},
        {"name": "zinc_06", "smiles": "N#Cc1c(Cn2cc(I)cn2)cn2ccccc12"},
        {"name": "zinc_07", "smiles": "O=c1ccoc2c(Cl)cc(Cl)cc12"},
        {"name": "zinc_08", "smiles": "COc1ccc(Nc2cc(C)ccc2N)cc1C"},
        {"name": "zinc_09", "smiles": "CC(=O)Nc1cc(C)c(N)cn1"},
        {"name": "zinc_10", "smiles": "O=[S@@]1CCCNc2ccccc21"},
        {"name": "zinc_11", "smiles": "N#Cc1c[nH]c2ccc(Cl)cc12"},
        {"name": "zinc_12", "smiles": "Cc1ncccc1NC(=O)OCCO"},
        {"name": "zinc_13", "smiles": "Cn1nc(Br)cc1N"},
        {"name": "zinc_14", "smiles": "CSCCc1ccc(N)cc1"},
    ],
}


def get_seeds(property_name):
    """[{name, smiles, score}, ...] for the given property, scored fresh."""
    raw = _RAW[property_name]
    scores = PROPERTIES[property_name]["score_batch"]([m["smiles"] for m in raw])
    seeds = []
    for m, s in zip(raw, scores):
        if s is None:
            continue
        seeds.append({"name": m["name"], "smiles": m["smiles"], "score": s})
    return seeds


if __name__ == "__main__":
    for prop in _RAW:
        seeds = get_seeds(prop)
        print(f"{prop}: {len(seeds)} seeds, best = {max(s['score'] for s in seeds):.3f}")
