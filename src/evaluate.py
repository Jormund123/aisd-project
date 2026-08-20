"""CLI tool: validate SMILES and compute ESOL LogS."""

import sys
from esol import validate_smiles, calculate_esol


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <SMILES>")
        sys.exit(1)

    smiles = sys.argv[1]
    mol, canonical = validate_smiles(smiles)

    if mol is None:
        print(f"INVALID SMILES: {smiles}")
        sys.exit(1)

    result = calculate_esol(smiles)
    print(f"Input:      {smiles}")
    print(f"Canonical:  {result['canonical_smiles']}")
    print(f"LogS (ESOL): {result['logs_esol']:.4f}")
    print(f"  LogP:      {result['logp']}")
    print(f"  MW:        {result['mw']}")
    print(f"  RotBonds:  {result['rotatable_bonds']}")
    print(f"  AromProp:  {result['aromatic_proportion']}")


if __name__ == "__main__":
    main()
