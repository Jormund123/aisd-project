All the questions that needs to be cleared. The answers must be present in the @presentation.md file. The current @presentation.md file is too complex.

Context 1: find the molecule with highest aqueous solubility (LogS) using an LLM, budget = 15 evaluations per run

Question 1: Why are we even doing this? What is the benefit of even solving this problem?
Question 2: Why is the budget 15 per run? Why not 10 or 50?
Question 3: What is LogS value?
Question 4: What is the aqueous solubility?

Context: 2: Oracle: ESOL (Delaney) computes LogS from structure. Cheap, R^2 ~0.69, an estimator not truth
Question 1: What is Oracle? What is ESOL? Who is Delaney?
Question 2: How is LogS calculated from structure?
Question 3: Compared to what is it cheaper?
Question 4: What is the number R^2 ~0.69?

Context 3: Dataset: AqSolDB (9,982 molecules). Source of seeds, pools, random baseline. NOT the simulator
Question 1: What is random baseline?
Question 2: What do you mean by "not the simulator"?
Question 3: What is the difference between seeds and pools?

Context 4: 13 fixed seed molecules give the LLM its only context. Bar to beat = methanol +0.208
Question 1: Why is methanol set as the bar?

Context 5:
• WHAT: the LLM is the inventor. Each round it sees tested molecules + ESOL LogS, proposes one new SMILES
• No math in the loop: the LLM both proposes and decides direction
• WHY: simplest possible LLM optimizer; if it has chemical intuition it shows here with nothing helping it
• HOW: build prompt -> paste -> validate SMILES (RDKit) -> reject mixtures -> ESOL score -> log -> repeat
Question 1: What are ESOL LogS?
Question 2: What are SMILES?
Question 3: Why even do this when we know that an LLM's answer is non-deterministic?
Question 4: What is RDKit?
Question 5: What is the parameter for rejecting mixtures?

Context 6:
Three prompt variants to test how much wording matters:
– A Chain-of-thought: 'analyze features, then design'
– B Pragmatic: A + soft ban on trivial tricks
– C Constrained: hard limits (MW<150, polar group, <=1 ring, single molecule)
Question 1: Why not directly do the constrained prompt?
Question 2: What are these constraints and how were they chosen?
Question 3: How were the inital seed of molecules chosen?

Context 7:
The per-run metrics (from analysis.py)
Question 1: What is Best, Improve, Success, Hit-Rate, Invalid, and Diversity? Why not simply show one parameter "accuracy" regarding molecules that are correct and that are not?

Context 8:
the LLM exploits the oracle
Unconstrained (Variant A), the LLM found ESOL has no chain-length penalty and no ceiling
• It added -CH(OH)- units forever (polyol chains), +0.28 LogS each, up to dodecitol +3.31 (chemically absurd)
• Replicated across Gemini, DeepSeek, ChatGPT -> a property of ESOL, not one model

Question 1: What are polyol chains?

Context 9:
• Best-so-far LogS vs iteration, 6 runs
– A (red/orange): climbs high via the unbounded polyol exploit
– B (blue/purple): runs DISAGREE, B1 late spike (salt), B2 honest plateau +0.56
– C (green/brown): smooth legit climbs, no spikes
• Take: unconstrained = exploit; soft ban = unreliable; hard rules = consistent
Question 1: What is the convergence graph used for? How do I read it?

Context 10:
• invalid_rate.png (shown): A 15%, B 9.1% (path-dependent), C 0%
– Strict rules also make the LLM more reliable
• diversity.png: A least diverse (0.345, repeats the trick); B/C more diverse
– Removing the easy exploit forces exploration
• comparison.png: all 3 variants beat random (+0.41); the +17.14 ceiling is an ESOL illusion (use +2.14 on the slide)

Question 1: I don't know how to read any of this graphs or figures. How can I? I don't even know what calibration is?

Context 11:
• WHAT: the LLM is a guesser. Fixed pool of 30 molecules; it predicts LogS (mu) + confidence (1-10) for each
• The LLM does NOT choose what to test. An acquisition function does
• WHY: A1 showed the LLM can invent but exploit. A2 asks a narrower question: can it put a reliable number on solubility?
• HOW the loop:
– LLM predicts all 30 -> confidence mapped to uncertainty sigma
– EI or UCB scores each and picks ONE -> ESOL scores it -> repeat
– EI = expected gain over best + uncertainty bonus; UCB = mu + kappa\*sigma
• WHY EI and UCB: the two standard acquisition functions; everything else held fixed
• CRITICAL: the random pool's best was only +0.21; 1 of 30 beat the seed
Question 1: What is LogS (mu) and confidence (1-10)?
Question 2: What is an acquistion function and how does it work?
Question 3: Why do we even need a 'reliable number' on solubility? What advantage does it give? What is the real world application of the results?
Question 4: What is uncertainty sigma? What is kappa and mu?
Question 5: What is EI and UCB? Why do we need it? Why not other parameters?
Question 6: What does it mean when someone says "random pool's best was only +0.21; 1 of 30 beat the seed?

Context 12: A2: regression accuracy (from analysis2.py)
MAE
0.82
Large miss on this scale
RMSE
1.13
Big misses present
Bias
+0.73
Guesses too HIGH (optimist)
Pearson r
+0.46
Weak-moderate; some signal
R squared
-1.25
WORSE than guessing the mean
Sign acc vs seed
0.78
Direction usually right
Confidence vs error
-0.0007
Confidence is meaningless

Question 1: What is regression accuracy? Why did we measure it? What are these different parameters?

Context 13:
• LLM prediction mu vs true ESOL LogS; dashed = perfect
– Dots scatter widely, most ABOVE the line (+0.73 bias)
– Loose upward drift = weak +0.46 correlation
– Worst: predicted +2.5 for a true -0.28
• Take: off-the-shelf LLM is a weak, over-optimistic numeric predictor
Question 1: What is prediction mu and why are even comparing it to true ESOL LogS?
Question 2: I don't know how to read calibration2.png?

Context 14:
• convergence2.png (shown): y-axis spans only +0.208 to +0.210, a sliver
– All runs hit the pool ceiling in 3-4 iters, then flat (nothing better exists)
• confidence_error2.png: mean-error line is flat, corr -0.0007
– LLM never used confidence below 5, even when wrong
• comparison2.png: random +0.41 'beats' EI/UCB +0.21 ONLY because random draws from all 9,982; unfair

Question 1: Why do we need to calculate all these values?
Question 2: What does it mean by convergence, confidence score and so on?
Question 3: What do their results mean?
Question 4: Again, why are we not just calculating basic accuracy?

Context 15:
• WHAT: the LLM never gives a number. It only judges duels: 'is A or B more soluble?'
• Many duels -> a ranking -> an acquisition picks the next molecule to ESOL-score
• WHY: A2 showed the LLM is a poor regressor, but LLMs compare better than they predict. Test that hypothesis
• HOW the loop:
– Select ~25 informative duels -> LLM answers A/B as JSON
– Bradley-Terry fit: P(i beats j)=sigmoid(u_i-u_j), gives utility + uncertainty
– UCB on utility picks one -> ESOL scores it -> repeat
• Pool STRATIFIED by ESOL -> real ceiling +1.14 (vs A2's dead +0.21)
• Validated with an --auto ESOL oracle + 26 unit tests BEFORE any LLM run

Question 1: Aren't we here to predict new molecules?
Question 2: What is the use of this process or experiment?
Question 3: How is everythign working here? What is a ranking and how is an acquisition working here?
Question 4: What is Bradley-Terry Fit? How does it work?
Question 5: Why does UCB on utility pick one? What is ESOL scoring?
Question 6: How is stratification being done?
Question 7: How is the validation working?

Context 16: A3: ranking accuracy + outcome (from analysis3.py)
Question 1: Why are we not using the same parameters so that we can compare Approach 1, 2 and 3? 1 has a different set of metrics in analysis.py and approach 2 and 3 have different?
Question 2: What is Best, Regret, Spearman, Kendall, Pairwise acc, Top-1 hit, NDCG@5?

Context 17:
A3 Figure: ranking_quality3.png (headline)
• Bradley-Terry utility vs true ESOL LogS, per run
– Points trend upward, tighter in run 2 (Spearman 0.71) than run 1 (0.53)
– Good but not perfect ranking, matches 0.71-0.80 pairwise accuracy
• Take: YES the LLM is a decent pairwise ranker, and better at ranking than at regression

Question 1: How to read this graph?
Question 2: What even is pairwise ranking?
Question 3: Ho are we able to comparing pairwise ranking to regression?

Context 18:
• Best-so-far vs iteration; real y-axis range now (stratified pool)
– Both runs climb from +0.21 seed and reach the +1.14 ceiling
– Gain happens in a few jumps (hence low success-rate)
• Take: with a proper pool, the PBO loop genuinely optimizes within budget
• Validation: --auto perfect ranker reaches ceiling fast; 40% noise degrades to Spearman ~0.59
Question 1: I don't udnerstand anything from it?
Question 2: How to read convergence3.png?

Context 19:
Approach
Best (mean/max)
Improve
Success
LLM quality (own metric)
A1 direct
1.94 / 3.31
+1.73
0.47
hit-rate 0.92
A2 regressor
0.21 / 0.21
+0.00
0.10
MAE 0.82, R2 -1.30, sign 0.78
A3 ranker
1.14 / 1.14
+0.93
0.10
pairwise 0.75, Spearman +0.62, NDCG 0.78
Question 1: What is best (mean/max)?
Question 2: What is LLM quality (own metric)?

Context 20:
• 1. A cheap descriptor oracle is exploitable (A1)
– The LLM optimizes ESOL exactly as written until hard constraints force real chemistry
• 2. The LLM is a poor regressor but a decent ranker (A2 vs A3)
– Regressor: R^2 negative, confidence meaningless. Ranker: 71-80% pairwise correct
– This is the central result: comparison is the strength, exact prediction the weakness
• 3. Search-space design is the real bottleneck (A2 vs A3)
– Same loop: a failure on a dead pool (A2), a success on a stratified pool (A3)
• Plus: one shared, unit-tested metrics module -> every number is validated, none hallucinated
Question 1: What is a "cheap descriptor oracle"?
Question 2: What is "Regressor: R^2 negative, confidence meaningless. Ranker: 71-80% pairwise correct"?
Question 3: What do you mean by search space design?

Context 21:
• Fix the oracle's exploitability: cap LogS / penalize MW, or use a stronger solubility predictor
• Fix A2's pool: re-run on an ESOL-stratified pool (proven in A3) for a fair EI-vs-UCB fight
• More runs: with API access, 5-10 seeds per setting + confidence intervals
• Better uncertainty: numeric range, multi-sample spread, or Bradley-Terry std_err instead of self-reported confidence
• Combine approaches: LLM proposes candidates (A1) -> rank by duels (A3) -> evaluate. Use the LLM where it is strong
• Fine-tuning on solubility data; scale A3 duels (more duels -> tighter ranking, likely raises top-1 hit)
Question 1: Why didn't we do it?
Question 2: Why cap LogS or penalize MW?
Question 3: What would be a stronger solubility predictor?
Question 4: What do you mean by numeric range, multi-sample spread, or Bradley-Terry std_err?
Question 5: How are we giving self-reported confidence?
Question 6: Should we combine A1 and A3 now?

All of these questiona re to be answered without coding snippets. You are to explain in extremely layman terms so that I understand.
The main thing is why didn't we use same metrics everywhere so that we could compare the approaches?
