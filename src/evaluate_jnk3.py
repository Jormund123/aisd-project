"""CLI tool: validate SMILES and compute the JNK3 oracle score."""

import sys

from jnk3_oracle import calculate_jnk3, validate_smiles


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_jnk3.py <SMILES>")
        sys.exit(1)

    smiles = sys.argv[1]
    mol, canonical = validate_smiles(smiles)

    if mol is None:
        print(f"INVALID SMILES: {smiles}")
        sys.exit(1)

    result = calculate_jnk3(smiles)
    print(f"Input:      {smiles}")
    print(f"Canonical:  {result['canonical_smiles']}")
    print(f"JNK3 score: {result['jnk3_score']:.4f}")


if __name__ == "__main__":
    main()
