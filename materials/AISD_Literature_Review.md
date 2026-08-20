# Literature Review: AI for Ranking-based Optimization

**AISD Lab Course, University of Bonn**

---

## Paper 1: Taking the Human Out of the Loop: A Review of Bayesian Optimization
*Shahriari et al., Proceedings of the IEEE, 2016*

This is the foundational reference. It frames Bayesian Optimization as a sequential decision-making loop for optimizing expensive black-box functions. The setup: you have an unknown function f(x), evaluations are costly, and you want to find the global optimum in as few queries as possible.

The two core ingredients are a **surrogate model** (typically a Gaussian Process) that approximates f and quantifies uncertainty, and an **acquisition function** that uses the surrogate's predictions and uncertainty to decide where to sample next. The paper covers the main acquisition functions: Probability of Improvement (PI), Expected Improvement (EI), Upper Confidence Bound (UCB), Thompson Sampling, and Entropy Search. The key insight across all of them is the exploitation-exploration tradeoff -- you want to sample where the predicted value is good AND where uncertainty is high.

**Relevance to the lab:** This gives you the full theoretical backbone for Approaches 2 and 3. Your lab replaces the Gaussian Process surrogate with an LLM. The acquisition function logic stays the same.

---

## Paper 2: Preferential Bayesian Optimization
*González et al., ICML, 2017*

Standard BO requires scalar observations (the exact value of f(x)). This paper relaxes that requirement. Instead, the optimizer only receives pairwise comparisons: "is x₁ better than x₂?" This is called a **duel**.

PBO models a latent preference function using a GP and uses a probit likelihood to convert pairwise comparisons into probabilistic feedback. The key finding is that modeling correlations between duels (not treating each comparison independently) drastically reduces the number of comparisons needed to find the optimum.

**Relevance to the lab:** This is the direct theoretical basis for Approach 3. Instead of a human providing pairwise preferences, your LLM acts as the pairwise ranker. You prompt it with two molecules and it says which is less toxic. The PBO framework then uses those comparisons to guide sampling.

---

## Paper 3: Bayesian Optimization for Controlled Image Editing via LLMs (BayesGenie)
*Cai et al., Findings of ACL, 2025*

This paper demonstrates LLMs and BO working together in practice, applied to image editing. BayesGenie uses an LLM to generate text prompts from user requirements, then feeds those prompts to a diffusion model. BO is used to automatically tune the inference parameters (specifically the Classifier Free Guidance weights) to maximize output quality. The system is model-agnostic, meaning it works across different LLMs (Claude 3, GPT-4) without retraining.

**Relevance to the lab:** This is a concrete example of the LLM-in-the-BO-loop paradigm. The domain is different (image editing vs. molecular optimization), but the architecture pattern is the same: LLM provides the intelligence, BO provides the systematic search strategy.

---

## Paper 4: Leveraging Large Language Models for Predictive Chemistry
*Jablonka et al., Nature Machine Intelligence, 2024*

This is the most directly relevant paper for a solubility problem. The authors show that GPT-3, fine-tuned on chemistry question-answer pairs, can predict molecular properties competitively with or better than dedicated ML models, especially in the low-data regime. Molecules are represented as text (SMILES strings, IUPAC names), and the LLM is prompted with natural language questions like "What is the solubility of [SMILES]?"

Key findings: (1) LLMs work surprisingly well as chemistry surrogates even with very few training examples. (2) The text-based interface means no feature engineering is needed. (3) Inverse design is possible by simply inverting the question ("What molecule has solubility X?").

**Relevance to the lab:** This directly validates the idea of using an LLM as your regression surrogate (Approach 2) and tells you how to format the prompts. For solubility specifically, you feed the LLM SMILES strings and known solubility values, then ask it to predict new ones.

---

## Paper 5: GimmBO: Interactive Generative Image Model Merging via Bayesian Optimization
*Liu, Ling, and Jacobson, arXiv, 2026*

GimmBO applies Preferential BO to the problem of merging image generation adapters (LoRAs). Users compare pairs of generated images ("which looks better?"), and PBO uses those preferences to navigate a high-dimensional weight space. The paper introduces a two-stage BO backend that handles the sparsity and constrained weight ranges typical in real-world adapter merging.

**Relevance to the lab:** Another concrete application of PBO (Approach 3), but in a creative domain rather than scientific. Shows that the pairwise comparison paradigm scales to high-dimensional spaces when combined with smart dimensionality reduction.

---

## Synthesis: How the Papers Map to Your Lab

| Lab Task | Primary Paper(s) |
|---|---|
| Understanding BO theory | Paper 1 (Shahriari) |
| Approach 1: LLM as direct optimizer | Paper 4 (Jablonka) for prompt format |
| Approach 2: BO + LLM regressor | Papers 1 + 4 |
| Approach 3: PBO + LLM pairwise ranker | Papers 2 + 5 |
| Seeing LLM + BO integration in practice | Paper 3 (BayesGenie) |

## Reading Priority for Solubility

1. **Paper 4 (Jablonka)** -- Read first. Directly shows how to prompt LLMs for molecular property prediction, including solubility-adjacent tasks. Gives you the SMILES-based prompt templates.
2. **Paper 1 (Shahriari)** -- Sections I-IV only. Understand the BO loop, GP basics, and the three main acquisition functions (EI, PI, UCB). Skip the advanced topics.
3. **Paper 2 (González)** -- Focus on the problem formulation and how duels replace scalar observations. The GP-probit model section is what you'll adapt.
4. **Papers 3 & 5** -- Skim for architecture patterns. Not essential reading, but useful for seeing how others wired LLMs into the BO loop.
