"""
JNK3 property oracle (TDC's Oracle('JNK3'): a random-forest classifier over ECFP
fingerprints, predicting probability of JNK3 inhibition). Same call shape as
esol.py's calculate_esol(smiles), so it drops into the same pipeline: validate,
score, record, repeat.

The score itself is computed by a persistent subprocess running oracle_env's
interpreter (see src/jnk3_worker.py for why: a legacy sklearn pickle incompatible
with this repo's main venv). That subprocess is started lazily and reused across
calls, since importing TDC costs a couple seconds.
"""

import atexit
import os
import subprocess
import sys

from rdkit import Chem

ORACLE_ENV_PYTHON = os.path.join(
    os.path.dirname(__file__), "..", "oracle_env", "bin", "python3"
)
WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "jnk3_worker.py")

_proc = None


def _worker():
    global _proc
    if _proc is None or _proc.poll() is not None:
        if not os.path.exists(ORACLE_ENV_PYTHON):
            raise RuntimeError(
                "oracle_env not found. Run scripts/setup_jnk3_oracle_env.sh first "
                "(one-time setup: installs the legacy sklearn/rdkit stack the "
                "pretrained JNK3 classifier needs)."
            )
        _proc = subprocess.Popen(
            [ORACLE_ENV_PYTHON, WORKER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        atexit.register(_proc.terminate)
    return _proc


def validate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return mol, Chem.MolToSmiles(mol)


def calculate_jnk3(smiles):
    """Validate + score one SMILES. Returns {'canonical_smiles', 'jnk3_score'} or
    None if invalid. jnk3_score in [0, 1], higher = more likely a JNK3 inhibitor."""
    mol, canonical = validate_smiles(smiles)
    if mol is None:
        return None
    proc = _worker()
    proc.stdin.write(canonical + "\n")
    proc.stdin.flush()
    raw = proc.stdout.readline().strip()
    if raw == "":
        return None
    return {"canonical_smiles": canonical, "jnk3_score": round(float(raw), 4)}


def calculate_jnk3_batch(smiles_list):
    """Score many SMILES in one round-trip (much faster than repeated calls).
    Returns a list aligned with smiles_list; invalid entries are None."""
    canon = [validate_smiles(s)[1] for s in smiles_list]
    proc = _worker()
    results = []
    pending = [c for c in canon if c is not None]
    for c in pending:
        proc.stdin.write(c + "\n")
    proc.stdin.flush()
    scores = {}
    for c in pending:
        raw = proc.stdout.readline().strip()
        if c not in scores:  # keep first score if a SMILES repeats in the batch
            scores[c] = round(float(raw), 4) if raw != "" else None
    for c in canon:
        if c is None or scores.get(c) is None:
            results.append(None)
        else:
            results.append({"canonical_smiles": c, "jnk3_score": scores[c]})
    return results


if __name__ == "__main__":
    for smi in sys.argv[1:]:
        print(smi, "->", calculate_jnk3(smi))
