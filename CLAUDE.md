# CONTEXT: AISD Lab Course -- AI for Ranking-based Optimization (Solubility)

## Project Overview

This is a university lab course project at the University of Bonn (AISD Lab Course). The topic is "AI for Ranking-based Optimization." The supervisor is reachable at s13kwija@uni-bonn.de (Discord: ktirta). Deliverables are a presentation and a report.

The core idea: use LLMs to find molecules with optimal aqueous solubility (LogS). The LLM proposes new molecules, and a computational simulator (ESOL via RDKit) verifies their solubility. The goal is to find the best-performing molecule within a limited evaluation budget.

## The Three Approaches to Implement

**Approach 1: LLM as Direct Optimizer**
- Show the LLM known (SMILES, LogS) pairs as context
- Ask it to propose a new molecule with better solubility
- Validate the SMILES with RDKit, run ESOL to get LogS
- Add result to known data, repeat
- No math, no acquisition function -- the LLM does everything
- This is the PRIMARY FOCUS. Build this end-to-end first.

**Approach 2: Bayesian Optimization with LLM as Regressor**
- Ask the LLM to predict LogS for a list of candidate molecules
- Also ask for confidence (1-10 scale) as uncertainty proxy
- Feed predictions + uncertainty into an acquisition function (Expected Improvement or UCB)
- Acquisition function picks the next molecule to test
- Run ESOL, record, repeat

**Approach 3: Preferential BO with LLM as Pairwise Ranker**
- Present pairs of molecules to the LLM: "Which is more soluble, A or B?"
- Aggregate pairwise comparisons into a ranking (Bradley-Terry model or win-count)
- PBO acquisition function picks the next molecule
- Run ESOL, record, repeat

## Critical Constraint: No API Access

The student does NOT have LLM API access (no budget for OpenAI/Anthropic API). All LLM prompting is done MANUALLY via ChatGPT or Claude chat interface. The student copies a prompt, pastes it, reads the response, and records results by hand.

This means:
- The Python code handles everything EXCEPT the LLM call
- Code should generate prompts (print them for the user to copy-paste)
- Code should accept LLM responses as input (SMILES string pasted back)
- Code should validate, evaluate, record, and update state
- The workflow is: script generates prompt -> user copies to LLM -> user pastes response back -> script processes it

## Technical Stack

### Simulator: ESOL via RDKit
- Delaney's ESOL method estimates aqueous solubility (LogS) from molecular structure
- Uses 4 descriptors: molecular weight, LogP, rotatable bonds, aromatic proportion
- Implementation reference: github.com/PatWalters/solubility (esol.py)
- Takes any valid SMILES string, returns LogS value
- Accuracy: R^2 ~0.69 (approximate, but fine for this project)

### Dataset: AqSolDB
- pip install datasets
from datasets import load_dataset
ds = load_dataset("maomlab/AqSolDB", name="AqSolDB")
- ~9,982 molecules with SMILES + experimental LogS
- Source: github.com/mcsorkun/AqSolDB
- File: curated-solubility-dataset.csv (or .tab)
- Role: provides few-shot examples for prompts + random baseline comparison
- NOT used as the simulator -- ESOL is the simulator

### SMILES Validation
- LLMs frequently generate invalid SMILES strings
- Use RDKit's Chem.MolFromSmiles() to validate
- Track invalid generation rate as a metric
- Canonicalize valid SMILES with Chem.MolToSmiles()

### Dependencies
- rdkit (conda install -c conda-forge rdkit)
- pandas
- matplotlib
- numpy
- scipy (for acquisition function math in Approach 2)

## What to Build (Python Scripts)

### 1. evaluate.py (Core utility)
- CLI tool: takes a SMILES string, validates it, runs ESOL, prints LogS
- Usage: `python evaluate.py "OCCO"`
- Output: validity status, canonical SMILES, LogS value

### 2. optimizer_approach1.py (Main interactive loop)
- Loads seed molecules (curated few-shot examples from AqSolDB)
- Maintains state: known_results dict {SMILES: LogS}, iteration count, budget
- Each iteration:
  1. Generates a prompt string (prints it for user to copy-paste to LLM)
  2. Waits for user input (the SMILES string the LLM proposed)
  3. Validates SMILES
  4. Runs ESOL
  5. Records result (appends to CSV log)
  6. Prints current best, iteration progress
  7. Loops until budget exhausted
- Saves full results to CSV when done

### 3. prompt_templates.py (Prompt generation)
- Contains 3 prompt variants:
  - Variant A: Minimal -- just data + "propose a molecule"
  - Variant B: Chain-of-thought -- "analyze patterns first, then propose"
  - Variant C: Constrained -- adds MW/structural constraints
- Function: generate_prompt(known_results, variant="B") -> str

### 4. esol.py (ESOL calculator)
- Adapted from PatWalters/solubility
- Function: calculate_esol(smiles) -> float (LogS)
- Should also return the molecular descriptors used

### 5. baseline.py (Random baseline)
- Loads AqSolDB
- Randomly samples N molecules, reports best LogS
- Repeats K times, outputs mean and std
- Also reports the actual best molecule in AqSolDB (ceiling)

### 6. analysis.py (Post-experiment visualization)
- Reads results CSV files
- Generates:
  - Convergence curve: best LogS vs. iteration number
  - Comparison across prompt variants (if multiple runs)
  - Invalid SMILES rate
  - Tanimoto similarity heatmap of proposed molecules (diversity measure)
  - Bar chart comparing Approach 1 vs. random baseline vs. AqSolDB best

### 7. For Approach 2 (later): acquisition.py
- Expected Improvement: EI(x) = (mu - f_best) * Phi(z) + sigma * phi(z)
  where z = (mu - f_best) / sigma
- UCB: UCB(x) = mu + kappa * sigma
- Takes LLM predictions (mu) and confidence scores (mapped to sigma)
- Returns which candidate to test next

### 8. For Approach 3 (later): pairwise.py
- Bradley-Terry model or simple win-count ranking from pairwise comparisons
- PBO acquisition function

## Seed Molecules (Few-Shot Examples)

Select 10-15 molecules from AqSolDB that:
- Span the full LogS range (very soluble to very insoluble)
- Have short, readable SMILES
- Are chemically diverse

Example (to be refined after loading AqSolDB):
- Ethanol: CCO, LogS = -0.77
- Phenol: Oc1ccccc1, LogS = -0.26  
- Naphthalene: c1ccc2ccccc2c1, LogS = -3.60
- Aspirin: CC(=O)Oc1ccccc1C(O)=O, LogS = -1.93
- Caffeine: Cn1c(=O)c2c(ncn2C)n(C)c1=O, LogS = -0.63

## Prompt Template (Approach 1, Variant B)

```
You are a computational chemist. Your task is to propose a novel molecule 
with the highest possible aqueous solubility (LogS).

Here are molecules I have already tested and their measured LogS values 
(higher = more soluble):

{numbered list of SMILES: LogS pairs}

First, analyze which structural features in the data above correlate with 
high solubility. Then use that analysis to design a new molecule.

Requirements:
- Output a valid SMILES string
- The molecule must be a real, synthesizable organic compound
- Do not repeat any molecule already tested
- Briefly explain your reasoning (2-3 sentences)

Your proposed molecule (SMILES):
```

## Evaluation Metrics

- **Simple regret:** Best LogS found after N iterations
- **Convergence speed:** Best-so-far LogS plotted over iterations
- **Invalid SMILES rate:** Fraction of LLM proposals that fail validation
- **Diversity:** Tanimoto similarity between proposed molecules (RDKit Morgan fingerprints)
- **Comparison vs. random baseline** from AqSolDB

## Experiment Design

- Budget: 15 evaluations per run
- Minimum 3 runs per prompt variant for statistical significance
- 3 prompt variants (A, B, C) = 9 runs total for Approach 1
- Record everything in CSV: iteration, prompt variant, LLM used, SMILES proposed, valid?, LogS, best so far, LLM reasoning

## Project Structure

```
aisd-ranking-optimization/
├── README.md
├── requirements.txt
├── data/
│   └── aqsoldb.csv          # downloaded AqSolDB dataset
├── src/
│   ├── esol.py              # ESOL solubility calculator
│   ├── evaluate.py           # CLI validation + ESOL tool
│   ├── prompt_templates.py   # prompt generation for all variants
│   ├── optimizer_approach1.py # interactive optimization loop
│   ├── baseline.py           # random baseline generator
│   ├── analysis.py           # plotting and evaluation
│   ├── acquisition.py        # EI/UCB for Approach 2 (later)
│   └── pairwise.py           # Bradley-Terry for Approach 3 (later)
├── results/
│   ├── approach1_variantA_run1.csv
│   ├── approach1_variantB_run1.csv
│   └── ...
└── plots/
    ├── convergence.png
    ├── comparison.png
    └── ...
```

## Key Papers (for reference, already reviewed)

1. Shahriari et al. (2016) -- BO theory, acquisition functions (EI, UCB, PI)
2. González et al. (2017) -- Preferential BO, pairwise comparisons, probit likelihood
3. Cai et al. (2025) -- BayesGenie, LLM + BO for image editing (architecture reference)
4. Jablonka et al. (2024) -- LLMs for predictive chemistry, SMILES/IUPAC prompting, few-shot works well
5. Liu et al. (2026) -- GimmBO, PBO for image model merging (Approach 3 reference)

## Build Priority

1. esol.py (get the simulator working)
2. evaluate.py (CLI tool for quick SMILES -> LogS)
3. prompt_templates.py (generate copy-paste prompts)
4. optimizer_approach1.py (the interactive loop)
5. baseline.py (random baseline for comparison)
6. analysis.py (visualization after experiments)
7. acquisition.py and pairwise.py (Approaches 2 and 3, after Approach 1 is complete)

## Style Notes
- No em dashes in any writing (use commas, parentheses, or periods instead)
- The student's name is Anand Karna
- Keep code clean, well-commented, and modular