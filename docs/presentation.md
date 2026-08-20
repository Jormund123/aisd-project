# Using AI to Find Water-Soluble Molecules (Presenter Script)

**Student:** Anand Karna | **Course:** MA-INF 4245, Lab AI for Scientific Discovery, University of Bonn

This is a read-aloud script for the presentation. It is written in plain language so I can present by reading it. It has 12 pages: 1 introduction, 3 pages for each of the three approaches, 1 page of conclusions, and 1 page of future work. Every number in here comes from actually running my code, nothing is made up.

## A quick word list (so the rest makes sense)

I will use a few terms again and again. Here is what each one means in plain words:

- **Molecule:** a chemical, like table salt or sugar. I write each molecule as a short text code called a **SMILES** string. For example, `CCO` is the code for ethanol (drinking alcohol).
- **Solubility / LogS:** how well a molecule dissolves in water. I measure it with a number called **LogS**. Higher LogS means it dissolves better. That is the thing I am trying to maximize.
- **ESOL:** a small formula that estimates a molecule's LogS just from its structure. Think of it as a cheap calculator that stands in for a real lab measurement. It is my "answer key."
- **LLM:** a large language model, the AI chatbot (ChatGPT, Claude, Gemini, DeepSeek). It is the "brain" I am testing.
- **Evaluation and budget:** every time I ask the ESOL calculator to score one molecule, that counts as one "evaluation." I only allow 15 evaluations per attempt. This budget is the whole challenge: find the best molecule with very few tries.
- **Run:** one complete attempt from start to finish, using the full budget of 15 evaluations.
- **Seed molecules:** a short starter list of known molecules with known scores that I show the AI, so it has examples to learn from.
- **Generate vs pick:** in all three methods the search produces brand-new molecules that nobody handed it. Methods 2 and 3 do this with a small trained network (a VAE, explained on Page 5) that turns molecules into map coordinates and back, so the search can invent a molecule by choosing a new coordinate.

I will define the rest (VAE, GP, acquisition function, Bradley-Terry) on the page where they first matter.

---
<!-- PAGE 1 of 12 -->

# Page 1: Introduction

## What am I trying to do

In simple words: I want to use an AI chatbot to discover the molecule that dissolves best in water, but I am only allowed 15 tries. The AI is the brain that suggests or judges molecules. The ESOL calculator is the answer key that tells me the true score of any molecule. My job is to see how good the AI is at this search, and to do it within the tiny budget of 15 tries.

## Why this problem is worth studying

Finding molecules that dissolve well in water is a real, expensive problem in drug design, agriculture, and materials. A medicine that does not dissolve cannot be absorbed by the body. Normally you test solubility in a lab, which is slow and costly. The question I am really studying is bigger than solubility: can an AI chatbot act as a smart search assistant that finds a good candidate in very few tries? Solubility is just a clean, cheap test-bed for that idea. The benefit, if it works, is faster and cheaper discovery of good molecules.

You might also ask why the budget is exactly 15 tries and not 10 or 50. The budget stands in for "real lab tests are expensive, so you only get a few." The exact number is a design choice, not a law. 15 is small enough to be a genuine challenge (you cannot just try everything), but large enough to show a learning curve (you can see the method improve over several steps). 10 felt too short to see a trend; 50 would make the budget so generous that even random guessing would look good, hiding the differences between methods. 15 is the sweet spot for showing behavior clearly.

## What is fixed for all three methods

- **The answer key (ESOL):** a formula that turns a molecule's structure into a solubility score. It is cheap and instant, so I do not need a real laboratory. It is not perfect (it is roughly 70 percent accurate), so I treat it as an estimate, not absolute truth. I say this openly as a limitation.
- **The molecule database (AqSolDB):** a public collection of about 10,000 real molecules with measured solubility. I use it for two things: to pick my starter examples, and to build a "random guessing" comparison. It is not the answer key; ESOL is.
- **The starter examples (seeds):** 13 fixed molecules I always show the AI. The chatbot has no memory between chats, so these examples are the only hints it gets. The best starter molecule is methanol, with a score of +0.208. That is the bar to beat.
- **How I talk to the AI:** I had no paid access to the AI, so I copied each question by hand into the chatbot and pasted the answer back. My Python code does everything else automatically (scoring, recording, plotting).

## More about ESOL, my answer key

An "oracle" is just my name for the answer key: the thing that tells me the true score of any molecule I test. My oracle is ESOL, a small formula that estimates solubility from a molecule's structure. John Delaney is the scientist who published this formula in 2004, which is why it is sometimes called Delaney's method. So: oracle = my scorer, ESOL = the specific formula I use as that scorer, Delaney = the person who invented it.

ESOL looks at four simple features of a molecule: its weight, how greasy or oily it is, how floppy it is (rotatable bonds), and how much of it is ring-shaped (aromatic). It combines those four numbers with fixed weights into one solubility score. It never physically dissolves anything; it just does arithmetic on the molecule's shape and composition. It is cheaper compared to a real laboratory measurement: measuring solubility for real means buying or making the molecule and running an experiment, which costs time and money. ESOL gives an estimate instantly on a computer for free.

R squared is a school-report-card grade for how well ESOL's estimates match real lab measurements, on a scale where 1.0 is perfect and 0 is useless. My R^2 ~0.69 means ESOL is roughly 70 percent of the way to perfect: good enough to be useful, but clearly an estimate, not the truth.

## More about AqSolDB and the seed molecules

The random baseline is my "dumb comparison." Instead of using any AI, I just pick molecules at random from the database and see how good the best one is. If my clever AI method cannot beat random picking, the AI is adding nothing. So the random baseline is the bar that any real method must clear to be worth the effort.

The database (AqSolDB) is only a source of example molecules and a source for the random comparison. It is not what scores my molecules. The scorer is ESOL. I keep these separate on purpose: the database provides molecules, ESOL provides the grades. Mixing them up would confuse where the numbers come from.

Seeds are the small starter list of example molecules I always show the AI at the beginning (13 of them), so it has something to learn from and so the search has a known-good region to start inventing around. All three methods use the seeds this way and then invent brand-new molecules from there; none of them picks from a fixed menu of options. I picked these 13 molecules from the database so they spread across the whole solubility range (from barely soluble to very soluble) and are simple, well-known chemicals with short names, giving the AI a fair, varied set of examples without accidentally handing it the answer.

The starting bar itself is whatever the best example the AI already has in front of it is, and among the seeds methanol scores highest at +0.208. So "beating the bar" means "inventing or finding something better than the best thing I already showed you." It is not special chemistry, just the highest score in the starter set.

## The three methods, from "AI does everything" to "AI just compares"

| Method | What I ask the AI to do | Who invents the next molecule to test |
| --- | --- | --- |
| 1. Direct Optimizer | "Invent a new, better molecule" | The AI itself |
| 2. AI as Predictor | "Guess the score of these molecules" | A math engine (BO) invents it; the AI only scores (Page 5) |
| 3. AI as Judge | "Just tell me, is A or B more soluble?" | A math engine (BO) invents it; the AI only compares (Page 8) |

The order is on purpose. Method 1 asks the AI to do everything. Method 2 asks it for an exact number. Method 3 asks only for a simple comparison. Each step demands less from the AI, so I can find out what the AI is actually reliable at. Importantly, in all three methods the molecule tested is brand-new (invented, not chosen from a fixed list). In Methods 2 and 3 the inventing is done by a principled math engine (Bayesian Optimization), and the AI's only job is to give opinions that steer it.

## Why I do more than one "run"

- **Why repeat at all:** the AI is not consistent. Ask it the same thing twice and you can get different answers, especially across different chatbots and different days. So a single attempt could just be luck. Repeating checks whether a result is real or a fluke.
- **How many:** Method 1 ran twice per prompt version (three times for Version C), Method 2 ran three to four times per decision rule, and Method 3 ran four times. Two is the smallest number that shows whether a behavior repeats; three or four starts to give a rough spread. I would have loved ten, but every single AI answer was typed by hand, so this was the practical limit. More runs is on my future-work list.

## Why I measure so many "metrics"

Just saying "the best score I found was X" is not enough to judge a method. I measure several things, and I built one shared, tested piece of code so every method is measured the same fair way. The metrics answer three plain questions:

1. **Did I find the best one available?**
2. **Was the AI actually good at its specific job?** (This is measured differently for each method, which I will explain.)
3. **Was the search efficient and well-behaved?** (How fast, how often it helped, how often the AI gave broken answers, how varied its ideas were.)

The rest of the talk walks through each method with these three questions in mind.

---
<!-- PAGE 2 of 12 -->

# Page 2: Method 1, the Idea

## What Method 1 is

Here the AI is the inventor. Each turn, I show it every molecule tested so far with its score, and I ask it to invent one brand-new molecule that should dissolve even better. I then check that the molecule is valid, score it with ESOL, add it to the list, and repeat. There is no extra math helping it. The AI both invents the molecule and decides which direction to explore.

## Why I built it this way

This is the simplest possible way to use an AI as a search engine. If the AI has real chemistry intuition, it should show up clearly here, because nothing else is helping it.

## How one turn works, step by step

1. My code writes the question, including the starter examples and everything tried so far.
2. I paste it into the chatbot and copy back the molecule it suggests (as a SMILES code).
3. My code checks the molecule is real chemistry, using a tool called RDKit. It also rejects "mixtures" (two separate molecules stuck together, like a salt), because those let the calculator be tricked. A broken or rejected suggestion counts as a wasted turn: I ask again, but the budget still ticks down.
4. My code scores the valid molecule with ESOL, saves it, and updates the best score so far.

RDKit is a free chemistry toolkit for the computer. I use it as a spell-checker for molecules: it tells me whether a SMILES the AI wrote is a real, valid molecule or nonsense, and it cleans up the molecule into a standard form. It also computes the features ESOL needs. A "mixture" is two separate molecules written as one answer (like a salt, which is really two pieces). In SMILES these show up as a dot character separating the pieces. So my code simply rejects any answer that contains that dot: if there is a dot, it is more than one molecule, and I throw it out. I did this because mixtures let the AI trick the scorer.

## Why do this at all if the AI's answers change every time

The AI being inconsistent is not a bug to hide; it is part of what I am measuring. If a behavior shows up across repeated runs and across different chatbots, it is real. If it changes every time, that instability is itself an important finding to report. So non-determinism is a reason to measure carefully, not a reason to avoid the experiment. That is exactly why I run each experiment more than once.

## Why I tried three different question styles

The loop was always the same. The only thing I changed was how I worded the question. I made three versions to see how much the wording matters:

- **Version A (Think first):** "First analyze what makes molecules soluble, then design one." This tests whether asking the AI to reason helps.
- **Version B (Be realistic):** Version A plus a gentle rule: "do not just cheat by repeating one trick, give me a realistic, different molecule." This tests whether a soft warning is enough to keep it honest.
- **Version C (Hard rules):** strict limits the AI must obey (keep the molecule small, include a water-friendly chemical group, at most one ring, and no mixtures). This tests whether firm guardrails work better than gentle ones.

You might ask why I did not just start with the constrained prompt (Version C). Because then I would never discover the interesting failures. Starting loose (Version A) is what revealed that the AI cheats the scorer, and starting medium (Version B) revealed that a gentle warning is unreliable. If I had jumped straight to strict rules, I would have gotten tidy results but learned nothing about how and why the AI misbehaves. The three versions together tell a story; Version C alone would just be the ending.

The strict rules in Version C were not arbitrary either: each one closes a specific loophole I had already watched the AI exploit. The size limit blocks the endless-chain trick, the single-molecule rule blocks the stuck-together-mixture trick, and the water-friendly-group rule nudges it toward genuinely soluble chemistry.

## One important setup decision

At first I showed the AI the lab-measured scores, but I grade with the ESOL calculator, and the two do not always agree. So the AI was aiming at one target while I was scoring with another. I fixed this by showing the AI the ESOL scores instead, so everyone is aiming at the same target. The downside I accepted: ESOL gets a few molecules in the wrong order, but at least the whole experiment is consistent.

## Why two runs here

For Version A, I ran it a second time using different chatbots, to check that the surprising behavior on the next page was not just one chatbot's quirk. For Versions B and C, the second run checked whether the behavior repeats. That is the payoff of running twice: it turns "this happened once" into "this is reliable" or "this depends on luck." I later added a third run for Version C specifically, since it was already the most promising variant, to see if its good behavior holds up a third time too.

---
<!-- PAGE 3 of 12 -->

# Page 3: Method 1, the Results

## The big finding: the AI cheats the calculator

When I gave the AI no hard rules (Version A), it discovered a loophole in the ESOL calculator. ESOL gives points for a certain water-friendly chemical group and never stops, with no penalty for the molecule getting huge. So the AI just kept bolting on more and more of that group, building a longer and longer chain, gaining points every time, forever. In Run 1 it reached a molecule called dodecitol at a score of +3.31, which is chemically ridiculous: real molecules do not keep getting more soluble without limit. The same trick showed up across Gemini, DeepSeek, and ChatGPT in Run 2, so it is a weakness of the calculator, not of one chatbot.

A polyol is a molecule with many water-friendly "-OH" groups (the same kind of group that makes sugar and glycerol sweet and water-loving). A polyol chain is what you get when the AI keeps adding more and more of these groups in a row, making a longer and longer molecule. ESOL rewards each added group and never penalizes the growing size, so the AI kept lengthening the chain to farm points, reaching dodecitol at +3.31.

Then it became a back-and-forth. Every time I closed one loophole with a new rule, the AI found the next loophole. It went from the endless chain, to sticking two molecules together (a "mixture"), and only once I blocked both in code (Version C) did it finally design honest, realistic molecules. The lesson: a cheap calculator can be cheated, so designing the goal carefully matters just as much as the AI.

## What I measure here, and what each number means in plain words

| Metric | In plain words |
| --- | --- |
| Best found | The highest score reached in the run |
| Improvement | How much better than the starter bar (+0.208) |
| Hit-rate | Out of all the molecules the AI suggested, what fraction actually beat the starter bar. Since the AI gives no prediction here, this is how I judge the quality of its ideas |
| Success rate | What fraction of turns actually set a new record |
| Invalid rate | How often the AI gave a broken or illegal answer |
| Diversity | How varied its molecules were. Lower number means more variety, higher means it kept repeating itself |

A note: two textbook metrics (called "regret" and "normalized score") do not apply here. They need a fixed list of options with a known best answer, but Method 1 invents from scratch with no fixed list, so there is no fixed "best possible" to measure against. I mark them as not-applicable rather than fake a number.

## The actual numbers (straight from my code)

| Version.Run | Best | Improvement | Success | Hit-rate | Invalid | Diversity |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| A.1 | +3.314 | +3.106 | 0.75 | 0.92 | 0.00 | 0.407 |
| A.2 | +2.184 | +1.976 | 0.40 | 1.00 | 0.30 | 0.284 |
| B.1 | +2.751 | +2.543 | 0.27 | 0.89 | 0.18 | 0.200 |
| B.2 | +0.563 | +0.355 | 0.50 | 0.83 | 0.00 | 0.186 |
| C.1 | +1.586 | +1.378 | 0.40 | 1.00 | 0.00 | 0.232 |
| C.2 | +1.258 | +1.050 | 0.50 | 0.90 | 0.00 | 0.243 |
| C.3 | +1.781 | +1.573 | 0.67 | 1.00 | 0.00 | 0.253 |

## Why the numbers look like this

- **Version A scores highest but is the least varied.** Once it found the loophole, it stopped being creative and just repeated the same trick. Big score, no imagination.
- **Version B's two runs disagree (one reached +2.75, the other only +0.56).** In one run the AI found a different loophole and spiked; in the other it stayed honest the whole time. So a gentle "please be realistic" warning sometimes works and sometimes does not. It depends on luck.
- **Version C is the most reliable, and a third run confirms it.** All three runs gave valid, realistic molecules every time (zero broken answers across all three), and almost every suggestion still beat the starter bar. The third run even set a new personal best for Version C (+1.781), and its diversity score (0.253) sits right in line with the other two, so this is not a fluke, it is a genuinely repeatable, well-behaved pattern.

---
<!-- PAGE 4 of 12 -->

# Page 4: Method 1, the Figures

My analysis code makes four pictures. For each, I will say what it shows, why it looks that way, and what to take from it.

A quick primer on reading a convergence graph in general: it is a simple line chart showing progress over time. The bottom axis is the turn number. The side axis is the best score found so far. Each line is one run. You read it left to right: a line going up means the method is finding better molecules as it goes; a flat line means no improvement. "Convergence" just means the line eventually flattens out because it has stopped improving.

## Figure: convergence.png (the climbing curve)

- **What it shows:** the best score so far, turn by turn, one colored line per run. The dashed line near the bottom is the starting bar (+0.21).
- **Why it looks this way:** Version A's lines shoot up steeply (the loophole). Version B's two lines disagree (one stays flat then suddenly jumps near the end, the other climbs gently and stops low). Version C's lines rise smoothly and level off honestly, with no sudden jumps, because the hard rules block the loopholes.
- **Take away:** with no rules the AI cheats and it does so reliably; a gentle warning is hit-or-miss; firm rules give steady, realistic results.

## Figure: comparison.png (bar chart)

- **What it shows:** the best score from each version, next to "random guessing" (+0.41) and the database's theoretical maximum (+17.14).
- **Why it looks this way:** all three versions easily beat random guessing, which means the AI is genuinely smarter than luck. The +17.14 bar is so tall it squashes the chart, but that number is fake, caused by the same calculator loophole. For the real slide I will replace it with the realistic maximum (+2.14).
- **Take away:** using an AI as a molecule designer clearly beats random guessing, and the calculator's lack of an upper limit shows up even at the database level.

## Figure: invalid_rate.png (how often the AI broke the rules)

- **What it shows:** the fraction of broken or illegal answers, averaged over each version's two runs.
- **Why it looks this way:** Version A 15 percent (loose wording leads to messy answers), Version B 9 percent (luck-dependent), Version C 0 percent (strict rules keep it tidy).
- **Take away:** firm rules do not only stop cheating, they also make the AI more reliable.

## Figure: diversity.png (how varied the ideas were)

- **What it shows:** how similar the molecules were to each other. Lower means more variety.
- **Why it looks this way:** Version A is the least varied because it repeated one trick; Versions B and C are more varied because, with the easy trick blocked, the AI had to explore real chemistry.
- **Take away:** removing the easy cheat actually makes the AI more creative.

A word on how to read the invalid-rate and diversity charts specifically: taller bars in invalid-rate mean more broken answers; lower numbers in diversity mean more variety. The comparison chart is just bars of the best score from each version next to random guessing. One word does not apply to Method 1: "calibration," meaning whether the AI's predicted numbers match the true ones. That idea only matters starting in Method 2, since Method 1 never asks the AI for a number, only for an invented molecule.

## Why the second runs mattered

The second runs are what let me make three confident statements: Version A's cheating repeats (so it is the calculator's fault), Version B's behavior is luck-dependent, and Version C's good behavior is reliable. I could not say any of that from a single run. That is exactly why I ran each twice, and a third Version C run afterward turned "reliable in 2 out of 2" into "reliable in 3 out of 3."

---
<!-- PAGE 5 of 12 -->

# Page 5: Method 2, the Idea

## What Method 2 is

Now the AI is a guesser, not an inventor. It plays what the task calls a **surrogate**: a cheap stand-in judge that scores molecules so a math engine can decide where to search. But unlike a naive setup, the molecules being tested are not chosen from a fixed list. They are invented fresh each round. The inventing is done by Bayesian Optimization working inside a trained network, and the AI's only job is to predict a solubility number for the molecules that engine proposes.

Three pieces make this work:

1. **The VAE (the molecule map).** A VAE is a small neural network I trained once on about 9,000 real molecules. It has two halves: an **encoder** that turns any molecule into a list of 64 numbers (think GPS coordinates on a map of all molecules), and a **decoder** that turns any coordinates back into a real molecule. The magic: nearby coordinates are similar molecules, so the search can step to a new coordinate and the decoder hands back a brand-new but valid molecule. This is how Method 2 invents molecules without ever writing chemistry by hand.
2. **The GP (the landscape guesser).** A GP (Gaussian Process) is a small statistical model that looks at a few molecules whose solubility we "know" (from the AI's guesses) and sketches the whole solubility landscape over the map, including how unsure it is at each spot. Near known points it is confident; far away it says "I don't know."
3. **The acquisition function (the search strategy).** A formula that reads the GP's landscape and its uncertainty and picks the next coordinate worth testing, balancing "climb where it looks tall" against "explore where I am unsure." I used the two textbook choices, **EI (Expected Improvement)** and **UCB (Upper Confidence Bound)**.

## Why I built it this way (and why a VAE instead of just prompting)

Method 1 showed the AI can invent, but also cheat. This is the classic **Bayesian Optimization** setup from the textbooks, except the usual statistical predictor is replaced by the AI's opinions. My supervisor asked for exactly this generative design: search a continuous coordinate space and decode new molecules, rather than pick from a menu. The reason for the VAE is that BO needs a smooth space it can do math in. You cannot nudge a molecule's text a little and stay valid, but you can nudge coordinates on the VAE map and always decode to a real molecule. The AI is still used the manual way (I paste molecules in and read its predictions back); the VAE only handles the molecule-to-coordinates translation and the inventing.

## How one round works

1. The engine encodes the molecules we already know into coordinates, and generates a batch of new candidate molecules by taking small steps from the best-known ones (decode, keep the valid and sensible ones).
2. I paste those new candidates into the AI and it predicts a LogS number for each. Those numbers train the GP.
3. The GP sketches the landscape and its uncertainty over the candidates; the acquisition function (EI or UCB) picks a small batch to actually test.
4. The ESOL calculator scores the picks (this uses up tries from the budget of 15). Record, and the results feed the next round.

## Why two decision rules, and three runs each

- **Why EI and UCB:** they are the two standard acquisition functions, and comparing them is part of the textbook question. Everything else is held identical. EI (Expected Improvement) picks the molecule expected to beat the current best by the most, with a bonus for uncertainty. UCB (Upper Confidence Bound) simply adds the predicted height and a bonus for uncertainty and picks the highest.
- **Why three runs each:** the AI's guesses change between sessions, so repeats check the result is not a one-time accident. I ran EI three times and UCB three times.

You might wonder why we even need this at all, what the real-world use is. Because real lab tests are slow and expensive, so you want a cheap model to steer you to good candidates in few tries. If an AI's opinions can guide a principled search to invent good molecules on a laptop, a lab only has to physically test the few winners. Method 2 asks whether today's chatbots are good enough to be that steering signal. The finding (their ordering is good even though their exact numbers are shaky) is the useful real-world answer.

## One honest setup note

The VAE was trained on limited data and CPU time, so its "wild" imagination (sampling far from known molecules) tends to produce nonsense chains. I keep the search close to known-good molecules and run a sanity filter that throws away absurd candidates (for example a 30-carbon chain that fools ESOL). This keeps the invented molecules realistic.

My first six runs all used the same random seed for candidate generation, so they explored a similar region and could not tell me whether the method is robust or just got lucky once. I later added one seed-varied run to each decision rule to test this directly, and it revealed something worth knowing before Page 6: with a genuinely different set of candidates, EI still found its way back to the same best molecule, but UCB completely struck out. That result is the real headline of the next page.

---
<!-- PAGE 6 of 12 -->

# Page 6: Method 2, the Results

## The headline: it usually invents molecules well above the bar, but not always

Six of my seven runs climbed from the starting bar (+0.208) to the same best invented molecule at **+1.219** (a small amide/urea molecule, `NC(=O)C(N)C(=O)NC(=O)NC=O`, that the VAE designed, not something from any list). Two of those six (one EI, one UCB) used the exact candidate pool as the rest; the other four (three EI, one UCB) each used their own, genuinely different pool of invented candidates and still found their way to that same molecule. That is strong evidence this is a real, robustly-reachable peak on the map, not a fluke of one shared setup.

The seventh run (UCB, with its own different candidate pool) is the honest exception: every single one of its 15 picks scored worse than the starting bar, so its best-so-far never moved off +0.208. That is the first genuine failure case in the dataset, and it is worth reporting exactly as it happened rather than only showing the six successes.

## Grading the AI's predictions

Method 2 is the method where the AI gives an actual number, so this is where I grade its prediction skill. "Guess" is the AI's predicted score; "true" is the real ESOL score of the molecule.

| Metric | In plain words |
| --- | --- |
| MAE | The average size of its mistake. Bigger is worse |
| RMSE | Like MAE, but punishes the occasional huge mistake more |
| Bias | Does it tend to guess too high (positive) or too low (negative) |
| Pairwise accuracy | Out of all pairs of molecules, how often it put the two in the correct order. This is the fair way to compare it to the ranker in Method 3 |

## The actual numbers

| Rule.Run | Best found | MAE | Pairwise accuracy |
| --- | :-: | :-: | :-: |
| EI.1 | +1.219 | 0.83 | 0.876 |
| EI.2 | +1.219 | 0.65 | 0.857 |
| EI.3 | +1.219 | 1.39 | 0.844 |
| EI.4 | +1.219 | 1.03 | 0.857 |
| UCB.1 | +1.219 | 0.48 | 0.867 |
| UCB.3 | +1.219 | 0.60 | 0.819 |
| UCB.4 | +0.208 | 1.32 | 0.657 |

Pooled across all seven runs: **best-found mean +1.075 (std 0.354)** and **pairwise accuracy mean 0.825 (std 0.071)**.

## What these numbers mean (the key explanation)

- **The AI is still a shaky number-predictor.** An average miss of about 0.5 to 1.4 on this scale is large, the same weakness seen in the exact-number task. Its raw solubility numbers should not be trusted at face value.
- **But its ordering is usually good.** Six of the seven runs score between 0.82 and 0.88 pairwise accuracy, far above the 50 percent of a coin flip. This is the fair yardstick I also apply to Method 3, so the two can be compared directly.
- **Why the search usually succeeds despite the shaky numbers:** BO does not need perfect numbers, only a roughly correct sense of which direction is better. The GP smooths out the noise, and the acquisition function keeps exploring, so the loop climbs to +1.219 in most runs.
- **EI reliably rediscovers the same peak, even from a different starting pool.** All four EI runs land on +1.219, including the one (EI.4) that used a genuinely different set of invented candidates. That is real evidence this molecule sits in a strong, easy-to-reach region of the map.
- **UCB.4 is a genuine failure, not noise.** Its pairwise accuracy (0.657) is the lowest of the seven, and every one of its 15 picks scored below the seed. UCB's exploration bonus (kappa = 2.0) pushes it to try more uncertain, less-anchored candidates than EI does; with an unlucky candidate pool, that exploration can spend the whole 15-evaluation budget without ever landing near the good region. This is exactly the kind of honest failure I could not see while every run shared the same candidate pool.

---
<!-- PAGE 7 of 12 -->

# Page 7: Method 2, the Figures

My analysis code makes four pictures (in `plots/approach2/`).

## Figure: convergence2_gen.png (the climbing curve)

- **What it shows:** the best score so far, evaluation by evaluation, one line per run, with the starting bar (+0.21) and the random baseline (+0.41).
- **Why it looks this way:** six of the seven lines start at the bar, then step up and settle at +1.219. The seventh (UCB.4) stays completely flat at the bar for all 15 evaluations.
- **Take away:** the generative loop reliably invents molecules well above the bar in most runs, but the one flat line is an honest reminder that "most" is not "always."

## Figure: topk2_gen.png (the optimization curve my supervisor asked for)

- **What it shows:** for each round, the average score of the best three molecules found so far. This is a smoother, less lucky way to see progress than "single best."
- **Why it looks this way:** for six runs it rises round by round as the set of good molecules found fills in, then flattens as the search converges. For UCB.4 it never rises at all.
- **Take away:** progress is steady and real for most runs, not one lucky hit, but the search does not always find that progress.

## Figure: calibration2_gen.png (how good are the raw numbers)

- **What it shows:** a dot per tested molecule. Left-to-right is the true score; up-and-down is the AI's guess. The dashed diagonal is a perfect guesser.
- **Why it looks this way:** the dots scatter loosely around the line, missing by roughly 0.5 to 1.4 depending on the run. They do drift upward to the right, which is the AI's decent sense of order.
- **Take away:** the AI is a rough number-predictor, but its ordering is usable, which is exactly what BO needs, most of the time.

## Figure: pairwise2_gen.png (is the AI a good ranker)

- **What it shows:** the pairwise ordering accuracy of the AI's predictions for each run, with the coin-flip line at 0.50.
- **Why it looks this way:** six bars sit between 0.82 and 0.88, comfortably above chance. UCB.4 sits noticeably lower at 0.657, still above the coin flip but a clear step down from the rest.
- **Take away:** even when its numbers are off, the AI usually orders molecules correctly most of the time, but the one weaker run shows that "usually" needs to be said out loud, not assumed.

## Why the extra seed-varied runs paid off

The first six runs all shared one candidate pool, so I could not tell if the method was robust or just lucky once. Adding one genuinely independent run per decision rule (a different random seed for candidate generation) answered that: EI proved robust, rediscovering the same peak from a completely different starting pool; UCB did not, striking out entirely on its independent run. That is a real difference between the two acquisition rules that the first six runs could not have shown me.

---
<!-- PAGE 8 of 12 -->

# Page 8: Method 3, the Idea

## What Method 3 is

This time the AI never gives a number at all. It is still a **surrogate** (a cheap judge that steers the search), and the search still invents brand-new molecules with the same VAE map and BO engine as Method 2. The only thing that changes is how the AI gives its opinion. Instead of a score, I show it two molecules at a time and ask the easiest possible question: "which one is more soluble, A or B?" I collect many of these head-to-head comparisons (I call them **duels**), turn them into a ranking, feed that ranking to the GP, and the engine invents the next molecule to test. Yes, Method 3 invents new molecules too, exactly like Method 2: both use the same VAE map and BO engine to do the inventing, and the AI is a surrogate that only gives opinions to steer it. The one difference is the kind of opinion, a number in Method 2 versus a duel here.

## Why I built it this way

Method 2 showed the AI's raw numbers are shaky. But people, and AIs, are usually better at comparing two things than at scoring one thing precisely. (It is easier to say "this coffee is better than that one" than to score each out of 100.) Method 3 tests that idea directly inside the same generative loop: if I only ask for comparisons, is the steering signal as good or better, while asking less of the AI?

## How one round works

1. **Generate candidates:** exactly as in Method 2, the engine invents new molecules by stepping around the best-known ones on the VAE map and decoding them.
2. **Pick the duels:** my code chooses a smart batch of pairs (each new candidate against the strongest molecules found so far, plus a random opponent). I keep the batch small so I can paste it by hand. There is also a small bootstrap of seed-versus-seed duels at the start, so the very first round already has a ranking to work with.
3. **Ask the AI:** it answers each duel with just A or B.
4. **Turn votes into a ranking, then a landscape:** I use a standard method called **Bradley-Terry**, which turns win-loss records into a strength score for each molecule, exactly like ranking sports teams. Those strength scores are the "heights" I feed to the GP, which then sketches the solubility landscape over the map and how unsure it is.
5. **Pick the next test:** a UCB acquisition (strength plus a bonus for uncertainty) picks a molecule; ESOL scores it (using tries from the budget of 15). Record, repeat.

The key point: the pairs never build a molecule directly. They only produce a ranking. From "feed the GP" onward, Method 3 is identical to Method 2; only the source of the height number differs (a direct number in Method 2, a Bradley-Terry rating here).

## Why I tested the whole system before involving the AI at all

This was my safeguard against fooling myself. Before any real AI run, I built an "autopilot" mode where the ESOL calculator answers the duels itself (and I can add random mistakes on purpose, to simulate a clumsy judge), plus unit tests for the GP and the Bradley-Terry math. It proves the whole pipeline (encode, generate, rank, fit, decode, score) works and climbs before any AI is involved. Only then did I trust the real runs.

## Why four runs

The AI's duel answers change between sessions, so repeats check that the ranking quality and final result hold up rather than being one lucky session. I have four real runs.

---
<!-- PAGE 9 of 12 -->

# Page 9: Method 3, the Results

## The headline: comparisons also steer the search above the bar

All three real runs invented molecules well above the starting bar (+0.208): +0.755, +0.901, +0.755, all clearing the random baseline of +0.408. So a search driven only by "which is more soluble, A or B?" also works, using the exact same generative engine as Method 2.

## Grading the AI's ranking

Method 3's job is ordering, so I grade how well its ranking matches the true ESOL order. "Its ranking" is the Bradley-Terry strength score; "true" is the real ESOL order. I use **pairwise accuracy** (out of all pairs, how often it put the two in the correct order) as the headline, because that is the exact same yardstick I applied to Method 2, so the two methods can be compared fairly.

| Run | Best found | Duels asked | Pairwise accuracy vs ESOL | Candidate pool |
| --- | :-: | :-: | :-: | --- |
| 1 | +0.755 | 80 | 0.799 | seed pool (default) |
| 2 | +0.901 | 88 | 0.841 | seed pool (default) |
| 3 | +0.755 | 80 | 0.780 | seed pool (default) |
| 4 | +0.755 | 80 | 0.717 | independent, different seed |

(Run 3's main log was later lost to a file mishap, but its final ranking file survived, so its final best-found and pairwise accuracy above are recovered and real, just without a full evaluation-by-evaluation trajectory to plot. Run 4 is the genuinely independent repeat I added afterward, on a different candidate-generation seed.)

## What these numbers mean (the key explanation)

- **The comparisons are good (72 to 84 percent correct):** the AI judges "which is more soluble" well above the 50 percent coin-flip line. Asking for a comparison rather than an exact number is a natural, reliable use of the AI.
- **Comparable to the regressor's ordering (about 0.83 in Method 2):** this is the fair head-to-head my supervisor asked for. When both surrogates are reduced to pairwise orderings, the regressor (Method 2) and the ranker (Method 3) land close together, roughly 0.83 versus 0.78. The AI understands solubility ordering about equally well whether it expresses it as a number or as a duel.
- **Three of the four runs landed on the identical best molecule, including one from a genuinely different candidate pool.** Runs 1 and 3 used the default candidate-generation seed; run 4 used a different one, verified from its trajectory to be a completely different set of invented candidates. All three independently converged on the same molecule (glycinamide, `NCC(N)=O`, +0.7555). That is real evidence of a robustly-reachable peak, echoing exactly what happened in Method 2's EI runs, not a shared-seed coincidence. Run 4's pairwise accuracy (0.717) is a bit lower than the others, which is its own useful signal, a harder or less lucky batch of duels this time, not a different final answer.
- **Why the best found is a bit below Method 2's:** the ranker gives a coarser signal (just A or B, no magnitude), so the GP landscape is less sharp and the search climbs a little less far in the same budget. More duels per round would likely close the gap.

## Why I wrote tests for the math

I wrote known-answer tests for Bradley-Terry and the GP (for example: a perfect ranking must score 1.0; the GP must pass exactly through points it was trained on). When I tell you the AI got 84 percent of comparisons right, I want to be certain that number is correct and not a bug. Nothing here is guessed.

---
<!-- PAGE 10 of 12 -->

# Page 10: Method 3, the Figures

My analysis code makes four pictures (in `plots/approach3/`). One note before these: the plots only show three lines (runs 1, 2, 4), not four. Run 3's final result is real and recovered (Page 9), but its step-by-step evaluation trajectory was lost along with its main log file, so there is nothing to draw a line from. The tables and pooled numbers do include all four runs; the pictures cannot.

## Figure: convergence3_gen.png (the climbing curve)

- **What it shows:** the best score so far, evaluation by evaluation, one line per run, with the starting bar (+0.21) and the random baseline (+0.41).
- **Why it looks this way:** all three lines start at the bar and step up, run 2 climbing highest (+0.90), runs 1 and 4 landing on the same lower peak (+0.76) despite coming from two different candidate pools. The climb happens in jumps as the ranking sharpens.
- **Take away:** a comparison-only search reliably invents molecules above the bar within the budget, across genuinely independent runs.

## Figure: topk3_gen.png (the optimization curve my supervisor asked for)

- **What it shows:** for each round, the average score of the best three molecules found so far.
- **Why it looks this way:** it rises round by round, most steeply in run 2, then levels off for all three.
- **Take away:** progress is real and steady, not a single lucky duel.

## Figure: ranking_quality3_gen.png (the most important picture here)

- **What it shows:** a dot per molecule. Left-to-right is its true ESOL score; up-and-down is the Bradley-Terry strength the AI's duels gave it. If the AI ranked well, dots rise from left to right.
- **Why it looks this way:** the dots trend upward in all three runs, most tightly in run 2, matching the 72 to 84 percent pairwise accuracy.
- **Take away:** this directly answers "is the AI a good judge of solubility?" Yes, clearly better than chance, in every run.

## Figure: pairwise3_gen.png (is the AI a good ranker)

- **What it shows:** the pairwise ordering accuracy for each run, with the coin-flip line at 0.50.
- **Why it looks this way:** all three bars sit between 0.72 and 0.84, well above chance, with run 4 the lowest of the three.
- **Take away:** this is the number I place next to Method 2's on the comparison page, so the regressor and ranker are judged on one fair axis.

## Why the autopilot test deserves a mention

Before the real runs, my autopilot (ESOL answering its own duels) proved the system climbs, and a deliberately clumsy judge does measurably worse. This shows the success in the real runs comes from the AI's judging quality, not from a bug or a lucky region, because the same code behaves correctly when driven by a known, controlled judge.

---
<!-- PAGE 11 of 12 -->

# Page 11: Conclusions

## All three methods, side by side

| Method | Best score (average / best run) | Improvement over bar | How good was the AI at its job |
| --- | :-: | :-: | --- |
| 1. Direct Optimizer | 1.92 / 3.31 | +1.71 | good ideas: 93 percent beat the bar |
| 2. AI as Predictor (generative BO) | 1.07 / 1.22 | +0.87 | ordering good: pairwise accuracy 0.83, but 1 of 7 runs failed entirely |
| 3. AI as Judge (generative PBO) | 0.79 / 0.90 | +0.58 | ordering good: pairwise accuracy 0.78 |

All three now invent brand-new molecules; none picks from a fixed list. Methods 2 and 3 share the same VAE map and Bayesian Optimization engine, and differ only in how the AI gives its opinion (a number versus a comparison).

Since I ran each method more than once, the first column reports two versions of its best score: the average best across its runs (mean) and the single best run (max). Method 1 shows 1.92 / 3.31, meaning its runs averaged a best of 1.92 and its luckiest run reached 3.31. Method 2 shows 1.07 / 1.22: six of its seven runs (including one from a genuinely independent candidate pool) landed on the same invented peak, but one run, using a different pool, found nothing better than the seed at all, which pulls its average down. Method 3 shows 0.79 / 0.90, since three of its four runs (two on the shared default pool, one on a genuinely independent pool) landed on the same peak (+0.76) and one run reached higher (+0.90). Showing both is honest: the mean says what to expect typically, the max says what is possible on a good day, and for Method 2 the mean now also reflects a real failure, not just luck.

## An honest caveat I will say clearly

The three methods still ask the AI to do different jobs, so I will not crown a single winner from one bar chart. Method 1 invents with the AI doing everything, so I grade the quality of its inventions (hit-rate: how often its ideas beat the bar). Methods 2 and 3 use a principled engine to invent and ask the AI only for opinions, a number in Method 2 and a comparison in Method 3, so I grade the AI on one fair axis for those two: pairwise accuracy against ESOL, since a number can be reduced to orderings and a comparison already is an ordering. That is why Method 2 and Method 3 are directly comparable on the AI's ordering skill (about 0.83 versus 0.78), while Method 1's free-invention skill is measured its own way.

Separately, there is a shared outcome metric that IS the same for all three: the best molecule found and how fast the search climbed (the top-k-so-far curve). So the honest summary: I compare the search outcome across all three, I compare the AI's ordering skill fairly between Methods 2 and 3, and I use a method-specific measure only for Method 1's inventing. I built one shared, tested piece of code (`analysis_common.py`) so the comparable numbers are computed identically.

## My four main findings

1. **A cheap calculator can be cheated (Method 1).** The AI optimizes exactly what I ask, finding loopholes (endless chains, stuck-together mixtures) until firm rules force real chemistry. Designing the goal well matters as much as the AI. The same lesson reappeared in the generative methods, where I needed a sanity filter to stop the VAE producing absurd chains. In other words, a cheap "descriptor oracle" (a scorer built only from simple structural features, like ESOL's weight and greasiness) is easy to cheat, because it does not capture real chemistry deeply, so a free-inventing search can farm its blind spots.
2. **Generative BO with the AI as a surrogate usually works, and independent runs prove it is not a fluke (Methods 2 and 3).** Most runs invented molecules well above the starting bar (+1.22 and up to +0.90) and above random guessing, inside a budget of 15. Better still, when I gave a run its own independent candidate pool instead of the shared default, it still rediscovered the same peak, in both Method 2's EI runs and Method 3's runs. That rules out "shared setup got lucky once" as the explanation. This is what the generative design made possible: my earlier design let Methods 2 and 3 only pick from a fixed list, which capped what they could find. The corrected design (my supervisor's) has them invent brand-new molecules by searching a trained coordinate map with Bayesian Optimization instead.
3. **But "usually works" is not "always works," and only independent runs revealed that.** One Method 2 run (UCB, with its own independent candidate pool) found nothing better than the seed across all 15 tries, the first genuine failure in the whole dataset. It would have looked identical to the six successes if I had only ever used the shared default pool. This is the clearest argument in the whole project for varying the random seed across runs rather than reusing one.
4. **The AI orders solubility about equally well as a predictor or a comparer (Methods 2 versus 3).** On the same fair yardstick, the regressor scored about 0.83 pairwise (7 runs) and the ranker about 0.78 (4 runs). Its raw numbers are unreliable (average miss of roughly 0.5 to 1.4), but its sense of "which is more soluble" is solid either way, in the runs where the search found a good region at all. That is the central result: trust the AI's ordering, not its exact numbers, but do not assume the search itself always succeeds.

## Why my measurement setup is itself worth mentioning

I built one shared, tested measuring tool (`analysis_common.py`) so Methods 2 and 3 are judged the same fair way, reducing both to pairwise ordering accuracy against ESOL. I separated "did it climb" (comparable) from "how good was the AI at its job" (the shared pairwise yardstick). Every number I show comes straight from running the code on the recorded runs, so none of it is invented.

---
<!-- PAGE 12 of 12 -->

# Page 12: Future Work

This list exists mostly because of two limits: time and no paid AI access. Every AI answer was typed and pasted by hand, so I could not afford dozens of runs or thousands of duels. And some items (retraining an AI on chemistry, training a much larger VAE) are whole projects of their own. So this is the set of good ideas I could not fit into the budget of time and effort I had.

## Make the calculator harder to cheat

ESOL has no upper limit and no penalty for huge molecules, which is what let Method 1 cheat and what made the VAE's wild samples produce absurd chains. Options to discuss with my supervisor: keep ESOL but report the loophole as a finding (my current plan), add a small size penalty, or switch to a stronger solubility model. Capping the score or subtracting points for large size would remove that free lunch and force realistic chemistry, without needing a separate sanity filter. A stronger solubility model would be one that captures real chemistry better than ESOL's four simple features, for example a machine-learning model trained on many measured solubilities, or a physics-based simulation. These are slower and more complex, but harder to cheat and more accurate, so they would make the whole test more trustworthy.

## Train a better VAE

My VAE was small and trained on limited CPU time, so its imagination far from known molecules is poor, which is why I had to keep the search local and filter out nonsense. A larger VAE trained longer would let the search roam the molecule map more freely and invent more diverse, more soluble candidates.

## Vary the generation seed for independent runs (started, worth finishing)

My first six Method 2 runs used the same candidate-generation seed, so they explored the same region and all landed on +1.22. I have since added one independent-seed run to each decision rule, and it was immediately worth it: EI confirmed the peak is robust, UCB revealed a genuine failure I could not have seen otherwise. The obvious next step is more of these, ideally 3 independent-seed runs per rule, so the failure rate and the spread of best-found scores can be reported with real statistics instead of one data point each.

## Do more runs for stronger statistics

A handful of runs shows whether a behavior repeats, but cannot give tight error bars. With paid AI access (so I am not typing by hand), I would do 5 to 10 runs per setting and report proper averages with ranges.

## Scale up the judging method

In Method 3 I kept the duels per round small so I could paste them by hand. With paid access I could ask far more comparisons, which would sharpen the Bradley-Terry ranking, give the GP a crisper landscape, and likely close the small gap to Method 2's best-found.

## Combine the methods

The natural combination: let Method 1's free inventing propose candidates, then let Method 3's comparing choose among them with duels inside the BO loop. This uses the AI where it is strong (inventing and comparing) and avoids where it is weak (exact numbers).

## Train the AI on chemistry

Everything here used off-the-shelf chatbots with no special training. Training one on solubility data could improve both its predictions (Method 2) and its judging (Method 3).

---

*End of script. The pictures are in the `plots/` folder; the exact numbers are in the `results/` folder, and everything can be regenerated by running my analysis code.*

