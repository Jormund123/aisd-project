# JNK3 property: implementation plan

New optimization target added alongside the existing ESOL/solubility property, sourced
from [TDC's JNK3 oracle](https://tdcommons.ai/functions/oracles/#c-jun-n-terminal-kinases-3-jnk3).
Same pipeline as before (LLM proposes/scores/ranks molecules, budget-limited search) --
only the data and scoring function changed. This doc covers what's set up, what's
deferred, and why.

## What JNK3 is, as a property

TDC's `Oracle(name='JNK3')` is a random-forest classifier over ECFP molecular
fingerprints, trained to predict the probability that a molecule inhibits JNK3 (a
kinase implicated in several diseases). Output is a float in `[0, 1]`, higher = more
likely an inhibitor. Unlike ESOL, there's no closed-form descriptor formula --
scoring means running a real ML model, and most random organic molecules score
exactly 0 (only structurally on-target molecules get nonzero probability). This
changes seed selection: the existing 13 ESOL seeds all scored 0.0 on JNK3 (no
gradient for few-shot prompting), so a fresh, JNK3-appropriate seed set was needed
(see below).

## The blocker: the oracle's model file doesn't load in this repo's venv

`Oracle('JNK3')` downloads a **scikit-learn 0.23 (2020) pickle**
(`oracle/jnk3_current.pkl`, ~11MB -- this download succeeded, it's the real "dataset
download" step). But this repo's main venv runs **Python 3.14 + scikit-learn 1.9**,
and scikit-learn changed its internal decision-tree node array format in **1.3**
(added a `missing_go_to_left` field for NaN support). Loading a pre-1.3 tree pickle
in >=1.3 raises `ValueError: node array from the pickle has an incompatible dtype`,
unconditionally -- there's no forward-compat shim for a jump this old.

Chasing a fix inside the main venv (matching sklearn version, cp314 wheel) turned
out to be a dead end: no sklearn version both old enough to read the pickle
(<1.3, i.e. <=1.2.2) and new enough to ship a `cp314` wheel exists -- 1.2.2's last
wheels target `cp38`-`cp311`. Monkeypatching the fix in-place isn't viable either:
`sklearn.tree._tree.Tree` is a compiled Cython extension type, not reassignable
from pure Python.

**Resolution: an isolated `oracle_env` venv**, built via
`scripts/setup_jnk3_oracle_env.sh`, pinned to the last stack that can both load the
pickle and run: Python 3.11 (Homebrew) + `numpy<2`, `scikit-learn==1.2.2`,
`rdkit==2023.9.5`, `PyTDC` (installed `--no-deps`, see below). This venv is used
**only** to run `Oracle('JNK3')`; the main venv stays untouched (still
torch/rdkit-modern/pandas as before).

Also worth flagging: plain `pip install PyTDC` pulls in a large amount of unrelated
weight -- `transformers`, `cellxgene-census`, `tiledbsoma`, `gget`, `accelerate` --
because the PyTDC package bundles single-cell genomics and HF model-serving
utilities alongside the chemistry oracles. None of that is needed just to call
`Oracle('JNK3')`. Both venvs install PyTDC with `--no-deps` and add back only the
handful of small packages the oracle's actual import chain touches (`fuzzywuzzy`,
`networkx`, `packaging`, `huggingface_hub`, `setuptools<81` for `pkg_resources`).

## Architecture: subprocess bridge, not a rewrite

`src/jnk3_oracle.py` (main venv) exposes `calculate_jnk3(smiles)` /
`calculate_jnk3_batch(smiles_list)` with the **same return shape** as
`esol.py::calculate_esol`. Internally it validates the SMILES with the main venv's
rdkit, then talks to a persistent `oracle_env` subprocess
(`src/jnk3_worker.py`, one SMILES in / one score out per line over stdin/stdout) so
TDC's ~2-3s import cost is paid once per run, not once per molecule. Everything
downstream -- optimizer loops, prompt templates, CSV logging -- calls
`calculate_jnk3` exactly like it calls `calculate_esol`; no approach-level code
needs to know an isolated venv exists.

## Dataset: ZINC250k, not AqSolDB

AqSolDB (solubility-labeled) has no bearing on JNK3. The standard starting
population for JNK3/GSK3B goal-directed generation benchmarks in the literature
(GCPN, MolDQN, MARS, and TDC's own generation benchmark) is **ZINC250k** --
`tdc.generation.MolGen('ZINC')`, ~249k drug-like SMILES, no property labels. Already
downloaded to `data/zinc.tab` (11.8MB) via `oracle_env` (the only place `tdc` is
installed with its `generation` module fully working).

`src/build_jnk3_seed_pool.py` samples N molecules from `data/zinc.tab`, scores them
via `calculate_jnk3_batch`, and writes `data/jnk3_scored_pool.csv` (used once to
hand-pick `src/seed_molecules_jnk3.py`; rerunnable with a different `--n`/`--seed`
if more/different seed candidates are ever needed). On a 3000-molecule sample: range
`[0.0, 0.31]`, 58% nonzero.

## Seeds

14 molecules hand-picked from the scored pool in `src/seed_molecules_jnk3.py`,
spanning `0.000` to `0.310`. They're anonymous ZINC catalog entries (no common
names, unlike ethanol/caffeine/etc. for ESOL), so they're numbered `zinc_01..14`.

## Done vs. deferred

**Done (all three approaches):**
- `src/jnk3_oracle.py`, `src/jnk3_worker.py` -- scoring bridge (batch-capable via
  `calculate_jnk3_batch`, used by approaches 2 and 3 to score whole picked batches
  in one round-trip instead of one subprocess exchange per molecule)
- `src/evaluate_jnk3.py` -- CLI, mirrors `evaluate.py`
- `src/seed_molecules_jnk3.py`, `data/jnk3_scored_pool.csv`, `data/zinc.tab`
- `src/approach1/prompt_templates_jnk3.py` + `optimizer_approach1_jnk3.py` --
  line-for-line the same loop as `optimizer_approach1.py`, property swapped
- `src/approach2/prompt_templates2_jnk3.py` + `optimizer_approach2_gen_jnk3.py` --
  same VAE+GP+EI/UCB engine as `optimizer_approach2_gen.py`, unmodified (GP
  standardizes internally, no property-scale tuning needed); only the oracle call,
  seed builder, and regression prompt (`predicted_jnk3` field, 0-1 range) changed
- `src/approach3/prompt_templates3_jnk3.py` + `optimizer_approach3_gen_jnk3.py` --
  same VAE+GP+Bradley-Terry+PBO engine as `optimizer_approach3_gen.py`, unmodified;
  only the oracle call, seed builder, and duel prompt changed
- `scripts/setup_jnk3_oracle_env.sh` -- reproducible one-time setup
- All three smoke-tested end to end (Approach 1: fake pasted LLM response, budget=1;
  Approaches 2/3: `--auto` mode, budget=3, JNK3 oracle standing in for the LLM)

**Prompt design note (all three approaches):** unlike ESOL, where solubility rules
(polarity, MW, aromaticity) are common chemistry knowledge any LLM already knows
well, JNK3 activity depends on specialized kinase-inhibitor SAR that 14 anonymous,
mostly-zero-scored few-shot examples can't teach by pattern-matching alone. All
three JNK3 prompts (direct-design, regression, and pairwise-duel) now include a
short domain hint on ATP-competitive kinase hinge-binding chemotypes (aminopyridine/
aminopyrimidine/indazole/quinazoline cores, moderate MW) to ground proposals/
predictions in real SAR instead of leaving the LLM to guess from sparse numbers
alone. This is the one deliberate deviation from a pure prompt-wording swap.

**Deferred:**
- Real experiment runs (budget=15, 3+ runs per approach/variant) and
  `analysis.py`-equivalent plots for JNK3 -- infra is ready and smoke-tested, no
  real runs done yet
- `docs/jnk3/` write-up section, once results exist

Approach 1 was finished first (not 2/3) because it's the simplest, most standalone
loop, and validating the whole scoring chain (oracle_env, subprocess bridge, seed
selection) against one approach before touching the VAE/GP-based ones limited risk
to the parts of the pipeline shared with the working ESOL results. Once that chain
was proven, approaches 2 and 3 followed directly since generative_bo.py, gp.py, and
pairwise.py are already property-agnostic.

## Folder restructure strategy

Everything above was added **additively** -- nothing existing for ESOL was moved,
renamed, or edited. That was deliberate: the ESOL results/plots/tex are close to
final for the report, and touching shared paths risks breaking something right
before writeup. The trade-off is a naming asymmetry worth resolving once a second
property has real results:

**Current state (asymmetric, ESOL implied by absence of a name):**
```
results/approach1/approach1_variantB_run1.csv      # ESOL (no property in the name)
results/approach1/jnk3/approach1_variantB_run1.csv # JNK3 (namespaced)
src/esol.py                                         # ESOL scorer, src/ root
src/jnk3_oracle.py                                  # JNK3 scorer, src/ root
```

**Target state, once JNK3 (or a third property) has real runs:**
```
results/<property>/approach{1,2,3}/...
plots/<property>/approach{1,2,3}/...
docs/<property>/approach{1,2,3}.tex
src/properties/{esol,jnk3}.py          # scorer modules, one shared interface
src/approach{1,2,3}/optimizer_*.py     # take --property {esol,jnk3} instead of
                                        # one optimizer file per property
```

Migration, when it happens, is mechanical and low-risk (renames + constant changes,
no logic changes): move `results/approach*/*.csv` under `results/esol/approach*/`,
same for `plots/`; move `src/esol.py` -> `src/properties/esol.py` and update the
handful of `from esol import ...` lines; collapse `optimizer_approach1.py` +
`optimizer_approach1_jnk3.py` into one file parameterized by `--property`. Not done
now because: (a) it touches every approach-1 file plus the not-yet-written
approach-2/3 JNK3 variants, (b) the ESOL side has finished results/figures already
referenced from `docs/*.tex`, and a path rename there is pure risk for zero payoff
until a second property's results actually need to sit next to them. Do this pass
right before writing the JNK3 section of the report, once approach 2/3 exist and
it's clear both properties are staying.
