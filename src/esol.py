"""
ESOL (Estimated SOLubility) calculator using Delaney's method.
Adapted from PatWalters/solubility.
"""

import math
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


ESOL_INTERCEPT = 0.16
ESOL_COEF = {
    "logp": -0.63,
    "mw": -0.0062,
    "rb": 0.066,
    "ap": -0.74,
}


def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    canonical = Chem.MolToSmiles(mol)
    return mol, canonical


def aromatic_proportion(mol):
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms == 0:
        return 0.0
    return aromatic_atoms / heavy_atoms


def calculate_esol(smiles):
    mol, canonical = validate_smiles(smiles)
    if mol is None:
        return None

    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ap = aromatic_proportion(mol)

    logs = (
        ESOL_INTERCEPT
        + ESOL_COEF["logp"] * logp
        + ESOL_COEF["mw"] * mw
        + ESOL_COEF["rb"] * rb
        + ESOL_COEF["ap"] * ap
    )

    descriptors = {
        "canonical_smiles": canonical,
        "logp": round(logp, 4),
        "mw": round(mw, 4),
        "rotatable_bonds": rb,
        "aromatic_proportion": round(ap, 4),
        "logs_esol": round(logs, 4),
    }

    return descriptors
