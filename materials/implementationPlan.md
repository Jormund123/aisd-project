# Implementation Plan: AI for Ranking-based Optimization (Solubility)

**AISD Lab Course, University of Bonn**

---

## Important Caveat Before Starting

There is an ambiguity in the slides that you should clarify with your supervisor. The slides say "computational tools to verify the performance of a **new design**." This could mean two things:

**Interpretation A (Generative):** The LLM proposes entirely new molecules (new SMILES strings) that don't exist in any dataset, and you verify them with a computational solubility estimator. This is harder but more ambitious. [We will go with this for now]

**Interpretation B (Search):** You have a fixed pool of molecules. The LLM helps you find the best one in fewer tests. "New design" just means "a design you haven't tested yet."

This plan assumes Interpretation A (generative), since that's what the slides most naturally suggest. But if your supervisor says Interpretation B, the plan simplifies -- you skip the SMILES validation step and replace the ESOL simulator with a dataset lookup.

---

## Phase 0: Environment Setup

### 0.1 Python Environment

Install:
- `rdkit` (molecular representation + ESOL solubility estimation)
- `pandas` (data handling)
- `matplotlib` (plotting results)

RDKit installation:
```
conda install -c conda-forge rdkit
```

### 0.2 The Simulator: ESOL via RDKit

The ESOL (Estimated SOLubility) method by Delaney (2004) computes LogS directly from molecular structure. It uses four molecular descriptors: molecular weight, LogP (octanol-water partition coefficient), number of rotatable bonds, and aromatic proportion.

Use the implementation from PatWalters/solubility on GitHub. The core function takes a SMILES string and returns a LogS value. This is your "experiment" -- every call counts against your budget.

**Critical:** ESOL is an estimator, not ground truth. It has known error margins (R^2 ~0.69 on AqSolDB). This is fine for the lab -- you're evaluating the optimization framework, not doing real drug discovery. But mention this limitation in your report.

### 0.3 The Dataset: AqSolDB

Download from github.com/mcsorkun/AqSolDB. The CSV contains ~9,982 molecules with SMILES strings and experimental LogS values.

**Role of AqSolDB in this project:** It is NOT your simulator. It provides:
1. Few-shot examples to include in your LLM prompts (so the LLM understands the task)
2. A reference pool to sanity-check whether the LLM is generating chemically reasonable molecules
3. Baseline data to compare against

### 0.4 SMILES Validation Utility

Since the LLM will generate SMILES strings, you need to validate them before feeding to ESOL. Write a small Python function using RDKit:

```python
from rdkit import Chem

def validate_smiles(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return False, "Invalid SMILES"
    return True, Chem.MolToSmiles(mol)  # canonical form
```

This is important because LLMs frequently generate invalid SMILES. You must track the invalid generation rate as a metric.

---

## Phase 1: Preparing the Few-Shot Examples

Before you start the optimization loop, you need a curated set of molecules with known solubility to show the LLM. This is the "training context" for in-context learning.

### 1.1 Selecting Seed Molecules

From AqSolDB, pick 10-15 molecules that:
- Span a wide range of LogS values (from very soluble to very insoluble)
- Have relatively short, readable SMILES strings (long SMILES confuse LLMs)
- Are chemically diverse (not all alcohols, not all aromatics)

Example selection:

| Name | SMILES | LogS |
|---|---|---|
| Ethanol | CCO | -0.77 |
| Phenol | Oc1ccccc1 | -0.26 |
| Naphthalene | c1ccc2ccccc2c1 | -3.60 |
| Aspirin | CC(=O)Oc1ccccc1C(O)=O | -1.93 |
| Caffeine | Cn1c(=O)c2c(ncn2C)n(C)c1=O | -0.63 |
| ... | ... | ... |

You'll refine this list during experimentation. The exact molecules matter less than coverage across the solubility range.

### 1.2 Representation Choice

Jablonka et al. found that IUPAC names often work as well as SMILES for LLM prompting. Test both:
- SMILES: compact, unambiguous, but LLMs sometimes garble them
- IUPAC names: human-readable, LLMs handle them well, but harder to feed into RDKit

Recommendation: Use SMILES as primary (since RDKit needs SMILES anyway), but include the common name where available for the LLM's benefit.

---

## Phase 2: Approach 1 -- LLM as Direct Optimizer (FULL DETAIL)

This is the simplest approach. The LLM does everything: it sees your data, reasons about chemistry, and proposes the next molecule to test. No acquisition function, no surrogate model. Just prompting.

### 2.1 The Optimization Loop

```
BUDGET = 15 tests
known_results = {seed molecules from Phase 1}

For each iteration (1 to BUDGET):
    1. Construct prompt with known_results
    2. Send prompt to LLM (manually, via ChatGPT/Claude)
    3. Record LLM's response (proposed molecule + reasoning)
    4. Validate the SMILES using RDKit
       - If invalid: record as failed attempt, re-prompt (does this count against budget? Decide upfront)
       - If valid: continue
    5. Run ESOL simulator on the valid SMILES
    6. Record the LogS value
    7. Add (SMILES, LogS) to known_results
    8. Track: best LogS found so far, cumulative results
```

### 2.2 Prompt Design

This is the most important part of Approach 1. The prompt must communicate:
- The task (find molecules with highest/lowest solubility)
- The known data (SMILES + LogS pairs)
- The output format (a single valid SMILES string)
- The direction (are you maximizing or minimizing LogS? Decide upfront)

**Template for maximizing solubility (finding most soluble molecules):**

```
You are a computational chemist. Your task is to propose a novel molecule 
with the highest possible aqueous solubility (LogS).

Here are molecules I have already tested and their measured LogS values 
(higher = more soluble):

1. CCO (ethanol): LogS = -0.77
2. Oc1ccccc1 (phenol): LogS = -0.26
3. c1ccc2ccccc2c1 (naphthalene): LogS = -3.60
4. CC(=O)Oc1ccccc1C(O)=O (aspirin): LogS = -1.93
5. [results from previous iterations...]

Based on your knowledge of chemistry and the patterns in this data, 
propose ONE new molecule that you predict will have a HIGHER LogS than 
any molecule tested so far.

Requirements:
- Output a valid SMILES string
- The molecule must be a real, synthesizable organic compound
- Do not repeat any molecule already tested
- Briefly explain your reasoning (2-3 sentences)

Your proposed molecule (SMILES):
```

### 2.3 Prompt Variations to Test

You should test at least 2-3 prompt variants to see which performs best. This is part of your experimental evaluation.

**Variant A: Minimal context.** Just the data, no explanation. Tests raw LLM chemistry knowledge.

**Variant B: Chain-of-thought.** Ask the LLM to first analyze what makes molecules in your set soluble, then propose. E.g., add: "First, analyze which structural features correlate with high solubility in the data above. Then use that analysis to design a new molecule."

**Variant C: Constrained exploration.** Add constraints like "The molecule should have molecular weight under 200" or "Focus on molecules with hydroxyl groups." Tests whether guided exploration helps.

### 2.4 Recording Results

Since you're prompting manually, rigorous record-keeping is essential. Create a spreadsheet (or CSV) with these columns:

| Iteration | Prompt variant | LLM used | SMILES proposed | Valid? | Canonical SMILES | LogS (ESOL) | Best so far | LLM reasoning (summary) | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | B | Claude | OCCO | Yes | OCCO | -0.52 | -0.52 | Ethylene glycol, two OH groups | Reasonable |
| 2 | B | Claude | OCC(O)CO | Yes | OCC(O)CO | 0.12 | 0.12 | Glycerol, three OH | Good improvement |
| 3 | B | Claude | C(=O)(O)C@@H... | No | - | - | 0.12 | Tried sugar, invalid SMILES | Re-prompted |

### 2.5 What to Measure

**Primary metric: Simple regret.** After N iterations, how good is the best molecule you found? Compare the best LogS across iterations.

**Secondary metrics:**
- **Convergence speed:** How quickly does the best-so-far improve? Plot best LogS vs. iteration number.
- **Invalid SMILES rate:** What fraction of LLM proposals are chemically invalid?
- **Diversity:** Are the proposed molecules structurally diverse, or does the LLM keep suggesting minor variations of the same thing? (Use Tanimoto similarity from RDKit to measure this.)
- **Prompt variant comparison:** Which prompt design leads to better results?

### 2.6 Baseline Comparisons

You need baselines to show Approach 1 is doing something useful:

**Random baseline:** Pick 15 random molecules from AqSolDB and report the best LogS among them. Repeat 10 times, report average. If the LLM can't beat random selection from a known dataset, something is wrong.

**Greedy baseline from dataset:** Sort AqSolDB by LogS and just pick the best. This is the ceiling for search-based approaches -- the LLM's generative proposals should ideally compete with or beat this.

### 2.7 Practical Workflow for One Session

Step by step, what you actually do:

1. Open your spreadsheet.
2. Open ChatGPT or Claude in a browser.
3. Copy-paste your prompt (with updated known_results from the spreadsheet).
4. Read the LLM response. Copy the SMILES string.
5. Open a terminal. Run your Python validation + ESOL script:
   ```
   python evaluate.py "OCCO"
   ```
   This script validates the SMILES and returns the LogS value.
6. Record everything in the spreadsheet.
7. Update the prompt with the new result. Go to step 3.
8. After 15 iterations, you're done with one run.

### 2.8 How Many Runs?

One run of 15 iterations is not enough for a convincing result. The LLM is stochastic. Run the full loop at least 3 times (with the same prompt variant) and report mean and variance of the best LogS found.

If time permits, run 3 times per prompt variant (A, B, C) = 9 total runs = 135 manual prompts. This is realistic for a lab course.

### 2.9 Things That Can Go Wrong (and How to Handle Them)

**Problem: LLM keeps generating invalid SMILES.**
Solution: Add explicit examples of valid SMILES format in the prompt. Or switch to IUPAC names and convert to SMILES yourself using RDKit or PubChem.

**Problem: LLM keeps proposing the same molecule or trivial variants.**
Solution: Add "Do not propose molecules with Tanimoto similarity > 0.8 to any previously tested molecule" to the prompt. Or explicitly list forbidden substructures.

**Problem: LLM proposes molecules that are technically valid but chemically absurd (e.g., a 500-atom chain).**
Solution: Add molecular weight constraints ("under 300 Da") and ring count constraints to the prompt.

**Problem: LLM "cheats" by proposing molecules it might have memorized from training data.**
Solution: This is actually fine for the lab. You're evaluating whether the LLM can guide optimization, not whether it's truly creative. Mention it as a limitation in your report.

**Problem: ESOL gives wildly different values than expected.**
Solution: Cross-check a few molecules against AqSolDB experimental values. If ESOL diverges a lot, note the discrepancy but proceed -- the optimization framework evaluation is still valid.

---

## Phase 3: Python Code You Need to Write

### 3.1 evaluate.py (Core utility)

Takes a SMILES string from command line, validates it, runs ESOL, prints the result. This is what you run after every LLM prompt.

### 3.2 analysis.py (Post-experiment)

Reads your results spreadsheet and generates:
- Best LogS vs. iteration plot (convergence curve)
- Comparison across prompt variants
- Invalid SMILES rate bar chart
- Tanimoto similarity heatmap of proposed molecules
- Comparison against random baseline

### 3.3 baseline.py (Random baseline generator)

Randomly samples N molecules from AqSolDB, reports the best LogS. Repeats K times. Outputs mean and std.

---

## Phase 4: Approach 2 -- BO with LLM as Regressor (BRIEF)

The key difference from Approach 1: you separate prediction from decision-making.

**Loop:**
1. Present the LLM with known results + a list of candidate molecules (either from AqSolDB or LLM-generated).
2. Ask the LLM to predict the LogS of each candidate. Record predictions.
3. Also ask for confidence (e.g., "On a scale of 1-10, how confident are you?"). This acts as your uncertainty estimate.
4. Manually compute the acquisition function (Expected Improvement or UCB) using the predictions and confidence scores.
5. Pick the candidate with the highest acquisition value.
6. Run ESOL. Record. Repeat.

**What you code:** The acquisition function. Expected Improvement is:
```
EI(x) = (mu(x) - best_so_far) * Phi(z) + sigma(x) * phi(z)
where z = (mu(x) - best_so_far) / sigma(x)
```
Where mu is the LLM's predicted LogS, sigma is derived from the LLM's confidence, Phi is the normal CDF, phi is the normal PDF.

**Challenge:** LLM confidence is not calibrated. A "confidence of 8/10" doesn't map to a real standard deviation. You'll need to experiment with how to convert LLM confidence into a usable sigma. This is a research question in itself -- discuss in your report.

**Key difference in prompting:** Instead of "propose a molecule," you ask "predict the LogS of THIS molecule." The LLM is a surrogate, not a decision-maker.

---

## Phase 5: Approach 3 -- PBO with LLM as Pairwise Ranker (BRIEF)

**Loop:**
1. Generate candidate molecules (from AqSolDB or LLM suggestions).
2. Present pairs to the LLM: "Which is more soluble, molecule A or molecule B?"
3. Collect pairwise comparison results.
4. Aggregate comparisons into a ranking (e.g., Bradley-Terry model or simple win-count ranking).
5. Use the PBO acquisition function to pick the next molecule to test.
6. Run ESOL. Record. Repeat.

**What you code:** The pairwise comparison aggregation (Bradley-Terry model or simpler). The PBO acquisition function (adapted from Gonzalez et al. 2017).

**Key difference in prompting:** The LLM never sees or produces a number. It only answers "A or B." This might be easier for the LLM to get right than exact LogS prediction.

**Challenge:** The number of pairs grows quadratically. With 20 candidates, that's 190 pairs. Manual prompting becomes very tedious. Consider batching: "Rank these 5 molecules from most to least soluble" instead of individual pairs.

---

## Timeline Suggestion

| Week | Task |
|---|---|
| 1 | Setup: Python env, RDKit, ESOL working, AqSolDB loaded. Select seed molecules. |
| 2 | Approach 1: Design prompts, run first 1-2 complete loops. Debug workflow. |
| 3 | Approach 1: Run all remaining loops (all prompt variants, 3 runs each). Generate baselines. |
| 4 | Approach 1 analysis. Begin Approach 2: design prompts, implement acquisition function. |
| 5 | Approach 2: Run loops. Begin Approach 3: implement pairwise aggregation. |
| 6 | Approach 3: Run loops. Full comparative analysis across all approaches. |
| 7 | Write report. Prepare presentation. |

---

## Open Questions for Supervisor

1. Is generative (LLM proposes new molecules) or search-based (LLM picks from a pool) the intended interpretation?
2. Which LLM should we use? Is there a preference for open-source vs. commercial?
3. Does an invalid SMILES generation count against the evaluation budget?
4. Is ESOL an acceptable simulator, or should we use a more accurate (but more complex) predictor?
5. How many optimization runs are expected for statistical significance?
6. Should we use the same LLM across all three approaches, or compare LLMs too?
