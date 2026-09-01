# AI for Ranking-based Optimization (Solubility / JNK3)

Uses an LLM to propose new molecules and searches for the best one under a limited
evaluation budget, in three ways:

- **Approach 1 (direct optimizer):** the LLM sees known (SMILES, score) pairs and
  proposes one new molecule per iteration. No math, no acquisition function.
- **Approach 2 (generative BO):** a SELFIES-VAE decodes new candidate molecules from
  latent space; the LLM predicts a score + confidence for each; a GP-based
  Expected-Improvement/UCB acquisition function picks which to evaluate.
- **Approach 3 (generative PBO):** same VAE + GP latent-space search, but the LLM only
  judges pairwise duels ("which molecule is better, A or B?"); a Bradley-Terry model
  turns the duels into a ranking that drives the acquisition function.

Two properties are supported end to end:

- **ESOL** (aqueous solubility, LogS) — closed-form RDKit descriptor calculation, no
  extra setup.
- **JNK3** (kinase inhibition probability) — a pretrained scikit-learn classifier, run
  in a separate legacy Python environment (see [JNK3 oracle setup](#jnk3-oracle-setup-optional)).

The LLM itself is never called via API. Each script prints a prompt, you paste it into
ChatGPT/Claude/Gemini, and paste the JSON response back. Every step besides the LLM call
(SMILES validation, scoring, logging, best-so-far tracking) is handled by the code.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`selfies_vae.pt` is already trained and committed under `models/`, so you don't need to
retrain it. If you ever do (e.g. after changing the VAE architecture):

```
python scripts/train_vae.py --epochs 12
```

### JNK3 oracle setup (optional)

Only needed if you want to run JNK3 experiments. The pretrained JNK3 model is an old
scikit-learn 0.23 pickle that this repo's main venv (modern sklearn) can't load, so it
runs in its own environment, invoked as a subprocess. One-time setup (requires
`brew install python@3.11`):

```
scripts/setup_jnk3_oracle_env.sh
```

ESOL needs none of this — it's a closed-form RDKit calculation.

## Running the approaches

Every run is identified by `--run <n>`, logs to its own CSV under `results/`, and takes
`--property {esol,jnk3}` (default `esol`) and `--budget <n>` (evaluations, default 15).
Add `--resume` to continue an existing run's CSV instead of overwriting it.

**Approach 1** — also takes `--variant {A,B,C}` (A = minimal, B = chain-of-thought,
C = constrained):

```
python src/_02_approach1.py --property esol --variant B --run 1 --budget 15
python src/_02_approach1.py --property jnk3 --variant C --run 1 --budget 15 --resume
```

**Approach 2** — also takes `--acq {ei,ucb}`:

```
python src/_03_approach2.py --property esol --acq ei --run 1 --budget 15
python src/_03_approach2.py --property jnk3 --acq ucb --run 2 --budget 15 --resume
```

**Approach 3**:

```
python src/_04_approach3.py --property esol --run 1 --budget 15
python src/_04_approach3.py --property jnk3 --run 1 --budget 15 --resume
```

Each of these prints a prompt block (`----- COPY THIS PROMPT -----`), waits for you to
paste the LLM's JSON response, validates/scores it, and loops until the budget is spent.

Add `--auto` to any of the three to skip the LLM entirely and let the property's own
oracle answer in its place — useful only for smoke-testing the pipeline end to end, not
for real results.

## Analysis and plots

Run analysis before plots — plots read the metrics CSVs analysis produces.

```
python src/_05_analysis.py --approach 1 --property esol
python src/_05_analysis.py --approach 2 --property esol
python src/_05_analysis.py --approach 3 --property esol
python src/_05_analysis.py --approach baseline --property esol --n 15 --k 1000

python src/_06_plots.py --approach 1 --property esol
python src/_06_plots.py --approach 2 --property esol
python src/_06_plots.py --approach 3 --property esol
python src/_06_plots.py --approach compare --property esol
```

Swap `--property esol` for `--property jnk3` to analyze/plot the JNK3 runs. Metrics land
in `results/approach<N>/[<property>/]metrics*.csv`; figures land in the matching
`plots/approach<N>/[<property>/]` folder.
