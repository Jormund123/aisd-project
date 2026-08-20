# Approach 1: LLM as Direct Optimizer (Presenter Notes)

**Student:** Anand Karna | **Course:** MA-INF 4245, University of Bonn

These are my speaking notes. They walk through what I did, in the order I did it, and the reason behind each step.

---

## What this approach is

Let me start with the goal. I wanted to use an LLM to find molecules with the highest aqueous solubility (LogS), within a budget of 15 evaluations per run. The way it works: the LLM proposes a molecule (as a SMILES string), and ESOL (my own code) scores it. I did not have LLM API access, so every prompt was copy-pasted by hand into ChatGPT, Claude, or Gemini.

For context, here are the three approaches in the whole project. Today I am presenting the first one.

| #   | Name             | LLM's job                   | Who picks next molecule       |
| --- | ---------------- | --------------------------- | ----------------------------- |
| 1   | Direct Optimizer | Propose a molecule          | LLM                           |
| 2   | BO + Regressor   | Predict LogS + confidence   | Acquisition function (EI/UCB) |
| 3   | Preferential BO  | Say "A more soluble than B" | Bradley-Terry ranking         |

In Approach 1 the LLM does everything: it proposes the molecule, and it decides what to try next. There is no extra math in the loop.

---

## Step 1: the setup I built first

Before any experiment, I set up the tools. The reason is that I needed a way to score molecules automatically, so the only manual step would be the LLM call itself.

- I made a **virtual environment** with rdkit, pandas, numpy, matplotlib, scipy, and datasets.
- I downloaded **AqSolDB** to `data/aqsoldb.csv`. It has 9,982 molecules with experimental LogS from -13.17 to +2.14 (the experimental value is in column `Y`). The reason I brought it in: it is my source of seed molecules and my random baseline. It is NOT the simulator.
- I implemented the **ESOL simulator** in `src/esol.py`. The formula is `LogS = 0.16 - 0.63*LogP - 0.0062*MW + 0.066*RotBonds - 0.74*AromProp`. The reason I used ESOL: it is the simulator named in the task, and it computes LogS from structure alone, so no lab is needed. Its accuracy is about R squared 0.69, so it is an estimator, not ground truth, and I note that as a limitation.
- I added **SMILES validation** with `validate_smiles()` in `esol.py` (RDKit MolFromSmiles plus canonicalization). The reason: LLMs often emit invalid SMILES, and I wanted to track the invalid rate as a metric.

---

## Step 2: the seed molecules I chose

Next I picked 13 fixed molecules with known LogS to show the LLM in every prompt. The reason I needed seeds: the LLM has no memory between sessions, so the seeds are its only task context. From them it infers structure-solubility patterns and can propose better molecules from the very first iteration.

I chose them by three criteria, and here is why each one matters:

1. Span the full LogS range, so the LLM sees both soluble and insoluble extremes.
2. Short SMILES (under 40 characters), because long SMILES confuse LLMs (Jablonka 2024).
3. Chemically diverse (amides, alcohols, ketones, aromatics, aliphatics), so the picture is not one-sided.

I hand-picked them from AqSolDB by common name so the LLM would recognize them. They live in `src/seed_molecules.py`.

| Name        | SMILES                       | LogS (ESOL) | Why included                       |
| ----------- | ---------------------------- | :---------: | ---------------------------------- |
| methanol    | `CO`                         |   +0.208    | High end: small polar OH           |
| acetamide   | `CC(N)=O`                    |   +0.114    | High end: amide H-bonding          |
| ethanol     | `CCO`                        |   -0.125    | High end: alcohol                  |
| acetone     | `CC(C)=O`                    |   -0.575    | High end: carbonyl                 |
| caffeine    | `Cn1c(=O)c2c(ncn2C)n(C)c1=O` |   -0.871    | Heteroaromatic, recognizable       |
| hexane      | `CCCCCC`                     |   -1.806    | Pure aliphatic, no polar groups    |
| cyclohexane | `C1CCCCC1`                   |   -1.836    | Aliphatic ring                     |
| aniline     | `Nc1ccccc1`                  |   -1.851    | Aromatic + amine                   |
| phenol      | `Oc1ccccc1`                  |   -1.935    | Aromatic + OH                      |
| toluene     | `Cc1ccccc1`                  |   -2.302    | Plain aromatic, contrast vs phenol |
| 1-decene    | `C=CCCCCCCCC`                |   -2.719    | Long hydrophobic chain             |
| naphthalene | `c1ccc2ccccc2c1`             |   -3.164    | Two aromatic rings                 |
| stilbene    | `C(=C/c1ccccc1)/c1ccccc1`    |   -3.890    | Low end: large flat hydrophobic    |

The values shown are ESOL values (I explain that decision next). The best seed to beat is methanol at +0.208.

---

## Step 3: a decision I had to make about the prompt numbers

Here is a subtle decision I made early. At first my prompts showed the experimental LogS (for example acetamide +1.581), but ESOL scores acetamide at +0.114. So the LLM was aiming at one scoreboard while I was grading on another.

The fix I chose: show the ESOL values in the prompts. The reason is that this gives one consistent objective, which makes the convergence curves interpretable. I re-sorted the seeds by ESOL.

The cost I accepted: ESOL misranks some molecules (it ranks hexane above phenol, which is chemically wrong). I accepted that, because the point is to optimize the ESOL surface consistently.

---

## Step 4: the three prompt variants I designed

The loop itself was the same every time. What I varied was the instruction in the prompt. All variants share the same seed block and the same JSON output format:

```json
{ "smiles": "OCCO", "name": "ethylene glycol", "reasoning": "..." }
```

Here are the three variants and the reason I made each:

- **Variant A (Chain-of-thought):** seed list plus "Step 1: analyze which features drive solubility. Step 2: design from that." I made this to test whether explicit reasoning helps.
- **Variant B (Pragmatic):** same as A, plus an explicit ban on trivial exploitation ("do NOT just lengthen a chain or repeat one motif; propose a realistic, distinct molecule"). I made this to test whether forcing chemical realism beats raw descriptor maximization.
- **Variant C (Constrained):** seed list plus hard limits (MW under 150, must have OH/NH2/C=O, at most 1 aromatic ring). I made this to test whether guardrails help.

A quick note on history: I originally had a "minimal" variant, but it behaved like chain-of-thought, so I dropped it. The old chain-of-thought variant became A, and the new B is the pragmatic one I wrote after Run 1 exposed the exploitation problem I describe below.

The full prompt text is in `docs/prompts.md`, generated automatically by `src/prompt_templates.py`.

On acceptance: a response counts if the SMILES is valid (RDKit accepts it). The quality of the reasoning does not affect the metric. An invalid response is a failed iteration: I re-prompt, but the counter still advances. I always run all 15.

---

## Step 5: the first run, and the key finding (ESOL exploitation)

When I ran Variant A, Run 1 (Gemini, 15 iterations), the best-so-far climbed every step, but it climbed through a degenerate path. The LLM discovered that ESOL has no chain-length penalty, so it just kept adding `-CH(OH)-` units.

| Iter | Molecule            | LogS (ESOL) | Best so far |
| :--: | ------------------- | :---------: | :---------: |
|  1   | urea                |   +0.403    |   +0.403    |
|  2   | glycerol (3 OH)     |   +0.772    |   +0.772    |
|  3   | ethanolamine        |   +0.517    |   +0.772    |
|  4   | ethylene glycol     |   +0.489    |   +0.772    |
|  5   | erythritol (4 OH)   |   +1.054    |   +1.054    |
|  6   | xylitol (5 OH)      |   +1.337    |   +1.337    |
|  7   | sorbitol (6 OH)     |   +1.619    |   +1.619    |
|  8   | heptitol (7 OH)     |   +1.902    |   +1.902    |
|  9   | octitol (8 OH)      |   +2.184    |   +2.184    |
|  10  | decitol (10 OH)     |   +2.749    |   +2.749    |
|  11  | decaethylene glycol |   -0.281    |   +2.749    |
|  12  | dodecitol (12 OH)   |   +3.314    |   +3.314    |

What I want you to take from this: the LLM exploits the simulator. From iteration 5 on it locks into the linear polyol (sugar-alcohol) series, adding one `CH(OH)` per step for a near-constant +0.28 LogS gain. ESOL rewards this forever (no solubility ceiling, no crystal-lattice penalty), so LogS rises without bound. Dodecitol at +3.31 is chemically absurd: real sugar alcohols do not keep getting more soluble forever. This is the core weakness of using a cheap descriptor model as the oracle, and it is exactly what motivated my Variant B, which explicitly forbids this move. I think this is a good result to report, because it shows the optimizer exploits the objective exactly as written, so objective design matters as much as the LLM.

(One housekeeping note: this run was labeled Variant B originally, and I relabeled it to Variant A after I restructured the variants. CSV: `results/approach1_variantA_run1.csv`.)

---

## Step 6: confirming it is not one model's quirk (Variant A, Run 2)

To check that the exploitation was not specific to one model, I ran a second Variant A, switching LLMs across iterations (Gemini, DeepSeek, ChatGPT).

| Iter | Molecule           |   LogS (ESOL)   | Best so far | Note                    |
| :--: | ------------------ | :-------------: | :---------: | ----------------------- |
|  1   | glycerol (3 OH)    |     +0.772      |   +0.772    |                         |
|  2   | urea               |     +0.403      |   +0.772    |                         |
|  3   | sorbitol (6 OH)    |     +1.619      |   +1.619    | chain restarts          |
|  4   | D-perseitol (7 OH) |     invalid     |   +1.619    | bad SMILES syntax       |
|  5   | glycine betaine    |     +0.547      |   +1.619    |                         |
|  6   | ethylene glycol    |     +0.489      |   +1.619    |                         |
|  7   | volemitol (7 OH)   |     +1.902      |   +1.902    | chain again             |
|  8   | octitol (8 OH)     |     +2.184      |   +2.184    | best                    |
|  9   | sodium gluconate   | REJECTED (salt) |   +2.184    | mixture guard caught it |
|  10  | choline chloride   | REJECTED (salt) |   +2.184    | mixture guard caught it |

This gave me two confirmations:

1. The polyol exploitation replicates across different LLMs. Gemini, DeepSeek, and ChatGPT all extended the sorbitol to volemitol to octitol chain. So it is a property of the ESOL objective, not of one model. The best single molecule was octitol at +2.18.
2. My no-mixture code guard works. When the LLMs tried salts (sodium gluconate, choline chloride) to exploit ESOL summing the fragments, both were rejected automatically at iterations 9 and 10. This is the same loophole Variant B hit, now closed in code for every variant.

CSV: `results/approach1_variantA_run2.csv`.

---

## Step 7: the headline story across all variants (the whack-a-mole)

Now let me tell the central story of Approach 1. Each prompt I wrote closed one exploit, and the LLM found the next one.

| Variant        | Best molecule found                   | LogS (ESOL) | Real discovery?                                          |
| -------------- | ------------------------------------- | :---------: | -------------------------------------------------------- |
| A              | dodecitol (12x OH chain)              |    +3.31    | NO. Infinite polyol chain, no solubility ceiling in ESOL |
| B              | Tris-myo-inositol complex             |    +2.75    | NO. Salt/mixture, ESOL sums disconnected fragments       |
| B (single mol) | Tromethamine (TRIS)                   |    +1.08    | YES. Compact, branched, realistic                        |
| C              | 2-amino-2-(aminomethyl)propanediamide |    +1.59    | YES. Compact, MW 146, all constraints obeyed             |

The finding I want to land: a cheap descriptor model like ESOL, used as the oracle, is exploitable. The LLM does not cheat maliciously; it optimizes the objective exactly as written. When I closed the chain loophole (A to B), it switched to mixtures. When I closed mixtures and capped size (C), it was finally forced into genuine, realistic chemistry. So my message is that objective and constraint design matter as much as the LLM.

A secondary observation I want to add: Variant C's tight box produced the most coherent search. The LLM followed a clear strategy (a quaternary-carbon core, swapping OH for NH2 for amide to add H-bonding) and explicitly respected the MW limit while reasoning. The guardrails turned blind descriptor-maxing into directed design.

---

## Step 8: why I built Variant C's constraints the way I did

Each constraint I added kills a specific ESOL exploit. They were chosen, not arbitrary, and here is my reasoning for each.

| Constraint                         | Purpose                                                                                                                                                                | Why not a different value                                                                                                                                                   |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MW < 150 Da                        | Core anti-exploitation lever. ESOL's MW penalty is tiny (-0.0062/Da), so a chain adds OH (gain) faster than MW (cost). A hard cap makes the infinite-chain trick impossible. | <100 too tight (kills valid amino-polyols). >200 lets sorbitol (182) and longer chains back in, reviving the exploitation. 150 is the sweet spot: drug-like small-molecule range. |
| Must have OH, NH2, or C=O          | Forces a polar H-bond group, the real physical driver of solubility. Steers off borderline hydrophobics.                                                               | Requiring all three kills too many valid molecules. "Any heteroatom" is too loose (ethers/halides do not drive solubility as strongly).                                     |
| <= 1 aromatic ring                 | One ring keeps drug-like realism; banning fused rings stops large flat hydrophobics (naphthalene -3.16, stilbene -3.89).                                               | 0 rings too restrictive (blocks legit aromatic soluble molecules). 2+ invites fused-ring hydrophobic exploitation.                                                                |
| Single connected molecule (no `.`) | Closes the Variant B iter-10 hole: a salt/mixture lets ESOL sum fragments for a fake high score. Now also code-rejected in the optimizer.                              | No looser option is safe; the prompt instruction alone failed in B, so it is enforced in code too.                                                                          |
| Pragmatic anti-trivial wording     | Carries B's lesson inside the box: no homologous-series padding.                                                                                                       | A constraints-only prompt (original C) was weak; it did not forbid the exploitation mindset, only the size.                                                                       |

---

## Step 9: why I also built baseline.py and analysis.py

A "best LogS found" number means nothing on its own, so I built two scripts to give it context and answer one question: is the LLM actually optimizing, or just getting lucky?

The first is **`baseline.py`, the control.** It answers "could random guessing do this too?" It draws 15 random molecules from AqSolDB (the same budget as the LLM), scores them with ESOL, keeps the best, and repeats 1000 times for a mean and standard deviation. The reason this matters: if the LLM does not beat this random best-of-15, then the LLM adds no value. It also reports the ceiling (the best ESOL in all of AqSolDB) so I know how far from the cap I am. Without a baseline I have no claim to make.

The second is **`analysis.py`, the evidence.** It turns the raw CSVs into the four plots I need. Here is how it uses the data:

1. **Convergence:** plots the `iteration` column against the `best_so_far` column for every run, starting from iteration 0 (the max ESOL score over the seed molecules).
2. **Invalid-rate per variant:** reads the `valid` column and computes `1 - mean(valid)` for the failure percentage (bad SMILES, corrupted text, or mixture rejections).
3. **Diversity (Tanimoto):** filters for valid rows, takes the `canonical_smiles`, converts them to Morgan fingerprints with RDKit, and computes the mean pairwise Tanimoto similarity.
4. **Comparison bar:** takes the maximum `best_so_far` across all runs for each variant, and plots those bars next to the `random_mean` and `esol_ceiling` from `results/baseline_summary.csv`.

In short: baseline.py is the yardstick, analysis.py draws the picture. Together they turn "I ran some prompts" into "here is the measured result and what it means."

---

## Step 10: the baseline numbers

I ran `baseline.py --n 15 --k 1000` and `analysis.py`. These numbers come straight from `results/baseline_summary.csv` and the PNGs.

| Quantity                                     | Value      |
| -------------------------------------------- | ---------- |
| Random best-of-15, mean (K=1000)             | **+0.408** |
| Random best-of-15, std                       | 1.377      |
| AqSolDB ESOL ceiling (max ESOL over dataset) | +17.14     |
| AqSolDB experimental ceiling (max Y)         | +2.138     |

Two things I want to flag honestly:

1. The random baseline is +0.408. That is the bar to beat. Every variant's best clears it (A +3.31, B +2.75, C +1.59), so the LLM does better than random sampling under the same 15-evaluation budget. Even the non-exploited results beat it (Variant C +1.59, and Variant B's best single molecule +1.08).
2. The ESOL ceiling of +17.14 is itself a degenerate ESOL value, not a real solubility. No molecule has LogS +17. It comes from ESOL's unbounded form (the positive rotatable-bond term and the lack of any cap) applied to some extreme AqSolDB entry. This is the same exploitation weakness as the polyol finding, now visible at the dataset level. The honest ceiling to quote is the experimental one, +2.138.

---

## Step 11: walking through the figures

I ran the experiment with 6 runs total (Variants A, B, C, two runs each). Let me walk through each figure.

### Figure 1: convergence.png

In this plot, each run has its own color: A1 red, A2 orange, B1 blue, B2 purple, C1 green, C2 brown. The dashed gray line at +0.21 is the best seed molecule, the starting point.

What we are looking at: this tracks the best LogS found so far across iterations, one curve per run (6 total).

Here is what happened, why, and what I take from it:
- **Variant A (red A1, orange A2):** Both climb high (A1 to +3.31, A2 to +2.18). They climb because there are no physical constraints, so the LLM found a mathematical loophole in ESOL: adding `-CH(OH)-` groups forever raises the score, so it built impossible, ever-longer polyol chains. The two runs reach different heights only because I stopped them at different iteration counts (12 vs 10), not because the behavior differs. What I learn: an unconstrained LLM ruthlessly exploits a simple math model. It optimizes the formula perfectly but ignores real chemistry, and it does so reproducibly across both runs.
- **Variant B (blue B1, purple B2), and the two runs DISAGREE, which is the key point:** B1 sits flat near +1.08, then spikes to +2.75 at the end. B2 climbs gently and plateaus low at +0.56, never spiking. The reason: the pragmatic prompt bans infinite chains, so B1 found a different loophole, a disconnected salt mixture whose fragment scores ESOL sums into a fake spike. B2, given the same prompt but a different LLM and trajectory, never found any loophole and stayed on honest zwitterion chemistry (urea to betaine to carnitine). What I learn: a soft "do not cheat" instruction is path-dependent. It sometimes works (B2) and sometimes fails (B1). It reduces exploitation but does not guarantee it away.
- **Variant C (green C1, brown C2):** Both rise smoothly and plateau legitimately (C1 +1.59, C2 +1.26), with no spikes. The reason: the hard physical constraints (MW under 150, single molecule, at most 1 ring) physically block both the infinite-chain and the salt-mixture loopholes, in both runs. What I learn: with hard rules the LLM behaves like a real chemist consistently, designing valid molecules that beat the seeds without exploitation.

### Figure 2: comparison.png

What we are looking at: a bar chart comparing the absolute best score found by each variant against random guessing ("Random") and the maximum possible score in the database ("ESOL ceiling").

Here is what happened, why, and what I take from it:
- **LLMs vs Random:** All three LLM variants (+3.31, +2.75, +1.59) easily beat the Random baseline (+0.41). The reason: the LLM is actively learning from the seed molecules and intelligently combining high-solubility functional groups, while random guessing is blind luck. What I learn: the core concept works. Using an LLM as a chemical optimizer is genuinely smarter and more effective than random search.
- **The giant ESOL ceiling (+17.14):** The ceiling bar is so huge it ruins the scale of the chart. The reason: this +17.14 is a mathematical illusion. Just like Variant A's chains, ESOL has no upper limit, so absurdly large molecules in the database get impossible scores. What I learn: the ESOL formula is fundamentally flawed at its extremes. (Note to self: in the final presentation I should replace this +17.14 bar with the experimental ceiling +2.14.)

### Figure 3: invalid_rate.png

What we are looking at: the percentage of times the LLM proposed something that broke the rules (invalid SMILES, weird text formats, or illegal salt mixtures). The bars are the mean over both runs of each variant.

Here is what happened, why, and what I take from it:
- **Variant A (15%):** A high failure rate (the mean of A1 at 0% and A2 at 30%). The reason: being unconstrained, it explored weird, large, or disconnected structures (bad SMILES in A2, plus salts the mixture-guard rejected). What I learn: loose prompts lead to messy, unpredictable outputs that break automated pipelines.
- **Variant B (9.1%):** A moderate failure rate, and it is the mean of two very different runs: B1 at 18.2% (it chased mixtures and corrupted text) and B2 at 0% (perfectly clean). The reason is the same path-dependence as Figure 1. What I learn: the pragmatic prompt's reliability is inconsistent run to run, which is another argument for hard constraints.
- **Variant C (0%):** A perfect 0% failure rate across both runs. The reason: the strict hard rules tightly focused the LLM. What I learn: strict constraints not only stop exploitation, they also make the LLM far more reliable, and they do it consistently.

### Figure 4: diversity.png

What we are looking at: how structurally different the proposed molecules are from one another, measured by Tanimoto similarity. A lower number means more diverse. Values are the mean over both runs of each variant.

Here is what happened, why, and what I take from it:
- **Variant A (0.345, least diverse):** The molecules look very similar to each other. The reason: once Variant A found the infinite-polyol-chain loophole, it stopped thinking and just added one more `-CH(OH)-` group each time. What I learn: reward hacking destroys creativity. If an AI finds an easy exploit, it will exploit it repeatedly rather than exploring new solutions.
- **Variant B (0.192) and Variant C (0.238), most diverse:** The molecules are highly varied. The reason: because I banned the simple chain exploit (in B) and added size limits (in C), the LLM was forced to search for different polar groups and different backbones. What I learn: forcing the LLM into a constrained box actually increases its chemical creativity and exploration.

---

## Step 12: what the second runs added

I decided that 2 runs per variant is enough (I do not need a third) to show the pattern is reproducible and not a single-path fluke. Here is what the second runs of B and C added.

### Variant B, Run 2 (Gemini + ChatGPT, 6 iterations)

| Iter | Molecule        | LogS (ESOL) | Best so far | Note                  |
| :--: | --------------- | :---------: | :---------: | --------------------- |
|  1   | urea            |   +0.4027   |   +0.4027   |                       |
|  2   | glycine         |   +0.3719   |   +0.4027   | zwitterion            |
|  3   | betaine         |   +0.5469   |   +0.5469   | zwitterion            |
|  4   | taurine         |   +0.2513   |   +0.5469   | sulfonate zwitterion  |
|  5   | TES (sulfobetaine) | -0.1986  |   +0.5469   | fixed a valence error |
|  6   | L-carnitine     |   +0.5626   |   +0.5626   | best                  |

The important finding: Variant B Run 2 never found an exploit. All 6 proposals were valid, connected single molecules (0% invalid). The LLM stayed on a coherent zwitterion theme (urea to glycine to betaine to taurine to carnitine) and topped out at a modest, fully legitimate +0.5626. This is the opposite of B Run 1, which found the salt-mixture loophole and spiked to +2.75. So the exploit is path-dependent, not guaranteed: the same pragmatic prompt can give either honest chemistry (Run 2) or a loophole (Run 1), depending on the LLM and the trajectory. The pragmatic ban makes exploitation less likely but does not eliminate it; only the hard constraints plus code guard (Variant C) reliably do. CSV: `results/approach1_variantB_run2.csv`.

### Variant C, Run 2 (ChatGPT + Sonnet + Opus, 10 iterations)

| Iter | Molecule                          | LogS (ESOL) | Best so far | Note               |
| :--: | --------------------------------- | :---------: | :---------: | ------------------ |
|  1   | serinol                           |   +0.7992   |   +0.7992   |                    |
|  2   | urea                              |   +0.4027   |   +0.7992   |                    |
|  3   | N-(2-hydroxyethyl)urea            |   +0.4989   |   +0.7992   |                    |
|  4   | 1,3-diamino-2-propanol            |   +0.8265   |   +0.8265   |                    |
|  5   | 2-amino-2-(aminomethyl)-1,3-propanediol | +1.1557 |   +1.1557   |                    |
|  6   | Tris                              |   +1.0807   |   +1.1557   |                    |
|  7   | 2-amino-2-(aminomethyl)-1,3-diaminopropane | +1.1829 | +1.1829   |                    |
|  8   | (duplicate of iter 7)             |   --        |   +1.1829   | duplicate, no score |
|  9   | 1,1,1,2-tetraaminoethane          |   +1.2579   |   +1.2579   | best               |
|  10  | guanidine                         |   +0.5255   |   +1.2579   |                    |

The finding: Variant C Run 2 replicates Run 1's behavior. All proposals obey the constraints (MW under 150, polar group present, at most 1 ring, single molecule). The LLM ran a directed search, packing small non-aromatic backbones with OH and NH2 groups, and ended at +1.2579 (versus Run 1's +1.5856). No exploitation, a smooth legitimate climb. This confirms the constrained variant produces realistic chemistry consistently, not by luck. CSV: `results/approach1_variantC_run2.csv`.

### Aggregate metrics across all 6 runs

| Variant | Best LogS (ESOL) | Invalid rate (mean of runs) | Diversity (mean pairwise Tanimoto) |
| ------- | :--------------: | :-------------------------: | :--------------------------------: |
| A       |      +3.3142     |            15.0%            |               0.345                |
| B       |      +2.7509     |             9.1%            |               0.192                |
| C       |      +1.5856     |             0.0%            |               0.238                |

Per-run best and invalid: A1 +3.314 (0%), A2 +2.184 (30%), B1 +2.751 (18.2%), B2 +0.563 (0%), C1 +1.586 (0%), C2 +1.258 (0%).

---

## Step 13: my three closing takeaways

To wrap up Approach 1, the second runs gave me three points to make:

1. **Reproducibility of the exploit (A):** the polyol chain showed up in both A runs, across Gemini, DeepSeek, and ChatGPT. The exploit is a property of the ESOL objective, robust across models.
2. **Path-dependence of the exploit (B):** B Run 1 found the mixture loophole, B Run 2 did not. A soft "do not cheat" instruction reduces but does not remove exploitation; the outcome depends on the LLM and the trajectory. This is my clean argument for hard constraints over soft wording.
3. **Reliability of constraints (C):** both C runs stayed legitimate, valid, and constrained (0% invalid both). Hard guardrails give consistent, realistic behavior, which is the main practical takeaway.

One open design question I want to raise for the supervisor: ESOL is exploitable. The options to discuss are to keep ESOL but report the exploitation as a finding (my current plan), add a cheap penalty term (for example cap LogS or penalize MW), or move to a stronger solubility predictor.

That concludes Approach 1. Approach 2 (BO with the LLM as a regressor) is in `docs/approach2.md`.
