# Approach 2: BO with the LLM as a Regressor (Presenter Notes)

**Student:** Anand Karna | **Course:** MA-INF 4245, University of Bonn

These are my speaking notes for Approach 2. I wrote them in plain language so anyone, even a beginner, can follow along. They walk through what I did, in order, and the reason for each step. (Approach 1 is in `docs/approach1.md`; the same ESOL simulator, AqSolDB dataset, and seed molecules carry over.)

---

## Step 1: what I set out to do, in simple words

Let me start by contrasting this with Approach 1. In Approach 1 the LLM was the inventor: it designed brand new molecules.

In Approach 2 I used the LLM as a guesser, not an inventor. I gave it a fixed list of 30 real molecules (I call this the "candidate pool") and asked it two things about each one:

1. **Predicted LogS**, which is the LLM's guess of how soluble that molecule is. I call this guess **mu**.
2. **Confidence**, a number from 1 to 10, which is how sure the LLM is about its own guess.

The important design choice here: the LLM does NOT choose which molecule I test next. A small piece of math, called an **acquisition function**, does that. I tried two acquisition functions:

- **EI (Expected Improvement):** it picks the molecule expected to beat my current best by the most, while also rewarding molecules the LLM is unsure about, because they might surprise me.
- **UCB (Upper Confidence Bound):** the simpler one. The score is the guess plus a bonus for uncertainty, and it picks the highest score.

Both of them turn the LLM's confidence into an "error bar" called **sigma**: high confidence means a small error bar (the LLM is saying "trust me"), and low confidence means a big error bar.

So the loop runs like this: the LLM guesses all 30, the acquisition function picks ONE, ESOL gives that molecule its true LogS (this is the real answer, and it costs me one of my limited evaluations), and then I repeat.

The reason I am doing all this, my research question, is: is the LLM a good enough guesser, and is its confidence meaningful, for this BO loop to find good molecules quickly?

---

## Step 2: what I ran

| Acquisition | Runs | LLMs used                  | Iterations each |
| ----------- | :--: | -------------------------- | :-------------: |
| EI          |  2   | Gemini (run1), ChatGPT + Sonnet (run2) | 10  |
| UCB         |  2   | Gemini (run1), ChatGPT + Sonnet + Gemini (run2) | 10 |

I used the same 30-molecule candidate pool every run (sampled from AqSolDB with a fixed random seed, so it is reproducible), and the same prompt every run. The only thing I changed was the acquisition function, EI versus UCB. The reason for keeping everything else fixed is that I wanted a clean comparison of the two acquisition functions. The code is in `src/acquisition.py`, `src/prompt_templates2.py`, and `src/optimizer_approach2.py`, and the plots are made by `src/analysis2.py`.

---

## Step 3: the first thing I checked, and the most important thing to understand

Before judging EI or UCB at all, I checked the candidate pool itself. I asked ESOL for the true LogS of all 30 pool molecules. Here is what I found:

- The best molecule in the pool was only **+0.2103** (N-(2-hydroxyethyl)acetamide).
- Only **1 out of 30** pool molecules was more soluble than my starting seed best (+0.208).
- The other 29 were equal or worse.

I want to stress why this matters so much. The pool was a random draw from AqSolDB, and most molecules in the real world (and in AqSolDB) are not very soluble. So by bad luck the pool had essentially one slightly-good molecule and nothing else. The best score any method could possibly reach on this pool was +0.21. No clever acquisition function can find a great molecule if no great molecule is in the list. Please keep this in mind as I go through every figure.

---

## Step 4: my results at a glance

| What | Value |
| ---- | ----- |
| Seed best (starting point) | +0.2080 |
| Pool ceiling (best possible on this pool) | +0.2103 |
| Best found by EI (both runs) | +0.2103 |
| Best found by UCB (both runs) | +0.2103 |
| LLM prediction error (MAE, average miss) | 0.82 LogS |
| LLM bias (does it guess too high or too low?) | +0.73 (guesses too HIGH) |
| Correlation between LLM guess and truth | +0.46 (weak-to-moderate) |
| Correlation between confidence and accuracy | -0.0007 (basically zero) |

The short version I would say out loud: every run found the pool's single good molecule, so in that narrow sense the loop worked. But the LLM was a noisy, over-optimistic guesser, and its confidence told me nothing about whether a guess was right.

---

## Step 5: walking through the figures

### Figure 5: convergence2.png

What we are looking at: the best score found so far after each iteration, one line per run (EI run 1 and 2, UCB run 1 and 2). The gray dashed line is the seed best (+0.208), the starting point. The green dotted line is the pool ceiling (+0.21), the best score possible on this pool.

I would ask the audience to read the y-axis carefully: it only spans +0.208 to +0.2103. That is a tiny range. The whole plot lives in a sliver, because, as I just explained, the pool offered almost no room to improve.

Here is what happened, why, and what I take from it:
- What happened: all four lines start at the seed best, sit flat for 2 to 3 steps, then jump up to the pool ceiling (+0.21) and stay there. EI reached the ceiling at iteration 3 in both runs. UCB reached it at iteration 4 in run 1 and iteration 3 in run 2. After that they are flat, because there was nothing better left to find.
- Why it happened: there was exactly one molecule above the seed best in the pool. Both acquisition functions found that one molecule within 3 to 4 tries, then could not improve further, which was impossible because nothing better existed.
- What I learn: EI and UCB behave almost identically here, and both are efficient at locating the single best item in a small pool. But this pool was too easy and too poor to tell EI and UCB apart in any meaningful way. The flat lines are not a failure of the method; they are the pool ceiling being hit.

### Figure 6: calibration2.png

This is the most informative figure for the real question of Approach 2: is the LLM a good guesser?

What we are looking at: a scatter plot. Each dot is one molecule the loop actually tested. The horizontal position is the molecule's true LogS (from ESOL). The vertical position is the LLM's guess (mu). The black dashed line is perfect prediction, where the guess equals the truth. A good regressor would have all dots sitting on that line.

Here is what happened, why, and what I take from it:
- What happened: the dots are scattered widely, and most of them sit above the dashed line. The average miss (MAE) was 0.82 LogS, which is large on this scale. The dots do drift up and to the right loosely (correlation +0.46), so the LLM is not random; it has some weak sense of which molecules are more soluble.
- Why the dots sit above the line: the LLM systematically guessed too high. On average its guess was +0.73 above the truth. It is an optimist; it thinks molecules are more soluble than they really are. The worst example: ChatGPT predicted +2.5 for a molecule whose true LogS was -0.28.
- What I learn: the LLM is a weak regressor for exact LogS values. It gets the rough ranking partly right (the weak positive correlation), but its absolute numbers are off by a lot and biased upward. This is the central honest finding of Approach 2: an off-the-shelf LLM, with no fine-tuning, is not a reliable numeric solubility predictor.

### Figure 7: confidence_error2.png

This figure tests the second half of my question: when the LLM says it is confident, is it actually more accurate?

What we are looking at: each dot is a tested molecule. The horizontal position is the confidence the LLM gave (1 to 10). The vertical position is how wrong its guess turned out to be (the absolute error). The black line connects the average error at each confidence level. If confidence were meaningful, that line would slope down (more confidence means less error).

Here is what happened, why, and what I take from it:
- What happened: the black average line is basically flat (it even rises a bit in the middle). The correlation between confidence and accuracy was -0.0007, which is essentially zero. Molecules the LLM rated 8 out of 10 confident were not meaningfully more accurate than ones it rated 5 out of 10. I would also point out that the LLM never used confidence below 5: it was always at least somewhat sure of itself, even when it was wrong.
- Why it happened: LLM confidence is uncalibrated. The model produces a confidence number because I asked for one, but that number is not connected to its real chance of being correct. This is a known issue with LLMs, and I flagged it in my plan as the open research risk.
- What I learn: the confidence score, which I fed into the acquisition function as the uncertainty (sigma), carried almost no real information. So the explore-versus-exploit balancing that EI and UCB are supposed to do was running on a meaningless uncertainty signal. This is a strong, reportable result: the uncertainty input to my BO loop was unreliable.

### Figure 8: comparison2.png

What we are looking at: a bar chart of the best LogS reached by EI, by UCB, by the random baseline, the pool ceiling, and the seed best.

I want to give an important warning here, so the audience does not misread the chart: the random baseline bar (+0.41) is higher than EI and UCB (+0.21). At first glance this looks like random beat my method. It did not, and here is why the comparison is unfair:

- The random baseline samples 15 molecules from the entire AqSolDB (all 9,982 molecules), so it can stumble onto soluble molecules that exist somewhere in that huge set.
- EI and UCB were only allowed to choose from my 30-molecule pool, whose ceiling was +0.21. They literally could not pick anything better, because nothing better was in front of them.

So the right comparison for Approach 2 is EI and UCB versus the pool ceiling (+0.21), not versus the whole-database random baseline. Against the pool ceiling, EI and UCB both reached it exactly. They did the best that was possible given what they were shown.

What I learn: the limiting factor in this experiment was the candidate pool, not the acquisition function. To make Approach 2 a fair fight, the pool must contain some genuinely soluble molecules.

---

## Step 6: my plain-language takeaways

To wrap up Approach 2, here are the four points I want to make:

1. **EI and UCB both worked correctly**, but they looked identical here, because the pool was tiny and had only one good molecule. Both found it fast. I cannot declare a winner between EI and UCB from this data.
2. **The LLM is a weak, over-optimistic regressor.** It gets the rough order partly right (correlation +0.46), but its exact LogS guesses miss by about 0.8 on average and lean too high (+0.73 bias).
3. **LLM confidence is meaningless here** (correlation with accuracy near zero). The uncertainty I fed the BO loop was unreliable, which undercuts the whole point of using confidence as sigma.
4. **The experiment was bottlenecked by the candidate pool**, not the math. A random pool from AqSolDB is mostly insoluble molecules, so there was almost nothing good to find.

I want to be clear that all four points are honest, reportable findings. Approach 2 is not a failure. It is a clear demonstration of where the LLM-as-regressor idea breaks down (numeric accuracy and confidence calibration), and a reminder that BO can only succeed if the search space contains good options.

---

## Step 7: what I would fix next, and where Approach 3 goes

For a fairer Approach 2 retest, if I have time, I would build the candidate pool so it actually contains soluble molecules, for example by sampling from the high-LogS end of AqSolDB, or using a larger pool, or letting the LLM propose candidates (which moves it closer to Approach 1). I would also try a better confidence-to-sigma mapping, although Figure 7 suggests the confidence signal itself is the problem.

My next planned work is Approach 3 (Preferential BO with the LLM as a pairwise ranker). Instead of asking the LLM for a number, I show it two molecules and ask "which is more soluble, A or B?" I collect many such comparisons, turn them into a ranking (a Bradley-Terry model), and let a preferential-BO acquisition function pick what to test next. The idea, and my reason for trying it, is that LLMs are usually better at comparing two things than at producing an exact number, so this may sidestep the weak-regressor problem I saw in Figure 6.

For reference, the code I added in this phase is `src/acquisition.py`, `src/prompt_templates2.py`, `src/optimizer_approach2.py`, and `src/analysis2.py`, and the plots are `plots/convergence2.png`, `calibration2.png`, `confidence_error2.png`, and `comparison2.png`.
