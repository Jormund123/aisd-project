"""
Long-lived JNK3 scoring worker. MUST be run with oracle_env's interpreter, not the
main venv: the pretrained JNK3 classifier (TDC's Oracle) is a scikit-learn 0.23
pickle, unreadable by the modern sklearn/rdkit stack the rest of this repo uses.
oracle_env pins numpy<2, scikit-learn==1.2.2, rdkit==2023.9.5 (see
scripts/setup_jnk3_oracle_env.sh) purely to keep that old pickle loadable.

Protocol: one canonical SMILES per line on stdin -> one float score per line on
stdout (flushed immediately), so jnk3_oracle.py in the main venv can drive this as
a persistent subprocess instead of paying TDC's ~2-3s import cost per call.
"""

import sys

from tdc import Oracle

oracle = Oracle(name="JNK3")

for line in sys.stdin:
    smiles = line.strip()
    if not smiles:
        continue
    try:
        score = float(oracle(smiles))
    except Exception:
        score = ""
    print(score, flush=True)
