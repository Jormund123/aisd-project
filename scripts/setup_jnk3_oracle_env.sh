#!/bin/bash
# One-time setup: builds oracle_env, an isolated venv that can load TDC's
# pretrained JNK3 classifier. That model is a scikit-learn 0.23 (2020) pickle;
# scikit-learn >=1.3 changed its internal tree node format and can no longer
# read it, and no Python new enough for this repo's main venv (3.14) has a wheel
# for an sklearn old enough to load it. So JNK3 scoring runs in its own venv
# pinned to the last compatible stack, invoked as a subprocess by
# src/jnk3_oracle.py. See docs/jnk3_implementation_plan.md for the full story.
#
# Requires Homebrew python@3.11 (brew install python@3.11).

set -euo pipefail
cd "$(dirname "$0")/.."

PY311="$(brew --prefix python@3.11 2>/dev/null)/bin/python3.11"
if [ ! -x "$PY311" ]; then
    echo "python@3.11 not found. Install it with: brew install python@3.11"
    exit 1
fi

"$PY311" -m venv oracle_env
source oracle_env/bin/activate
pip install -q --upgrade pip
pip install -q --only-binary=:all: \
    "numpy<2.0" "scikit-learn==1.2.2" "rdkit==2023.9.5" "pandas<2.2" \
    fuzzywuzzy requests tqdm networkx packaging "setuptools<81" \
    "huggingface_hub<1.0,>=0.20.3"
pip install -q PyTDC --no-deps

echo "oracle_env ready. Verifying JNK3 oracle loads..."
python -c "
from tdc import Oracle
o = Oracle(name='JNK3')
print('JNK3 oracle OK, CCO ->', o('CCO'))
"
echo "Done. src/jnk3_oracle.py will use this venv automatically."
