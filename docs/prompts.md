# Approach 1 Prompts (ESOL seed values)

Copy one block at a time into the LLM. Paste the JSON response back.

Seed block (identical in all variants):
```
 1. methanol       CO                               LogS = +0.208
 2. acetamide      CC(N)=O                          LogS = +0.114
 3. ethanol        CCO                              LogS = -0.125
 4. acetone        CC(C)=O                          LogS = -0.575
 5. caffeine       Cn1c(=O)c2c(ncn2C)n(C)c1=O      LogS = -0.871
 6. hexane         CCCCCC                           LogS = -1.806
 7. cyclohexane    C1CCCCC1                         LogS = -1.836
 8. aniline        Nc1ccccc1                        LogS = -1.851
 9. phenol         Oc1ccccc1                        LogS = -1.935
10. toluene        Cc1ccccc1                        LogS = -2.302
11. 1-decene       C=CCCCCCCCC                      LogS = -2.719
12. naphthalene    c1ccc2ccccc2c1                   LogS = -3.164
13. stilbene       C(=C/c1ccccc1)/c1ccccc1          LogS = -3.890
```

Note: the optimizer script (`optimizer_approach1.py`) generates these prompts
automatically with the growing tested list, so you normally do not copy these by
hand. They are kept here for reference and for manual one-off prompting.

---

## Variant A: Chain-of-Thought

```
You are a computational chemist optimizing aqueous solubility.

Below are molecules that have been tested, with their LogS values (higher = more soluble in water):

[seed block]

Step 1: Analyze the data above. Which structural features correlate with HIGH solubility? Which correlate with LOW solubility?

Step 2: Using that analysis, design ONE new molecule predicted to have a HIGHER LogS than any molecule above.

Requirements:
- Must be a real, synthesizable organic compound
- Do not repeat any molecule already listed
- Respond ONLY with valid JSON, no other text

{
  "smiles": "<valid SMILES string>",
  "name": "<common or IUPAC name>",
  "reasoning": "<reasoning>"
}
```

---

## Variant B: Pragmatic (anti-trivial)

Same as A, but explicitly forbids exploiting the ESOL formula by lengthening chains.
Added because Variant A degenerated into an infinite polyol series
(erythritol -> xylitol -> sorbitol -> ... -> dodecitol), which inflates ESOL LogS
without being a real discovery.

```
You are a computational chemist optimizing aqueous solubility.

Below are molecules that have been tested, with their LogS values (higher = more soluble in water):

[seed block]

Step 1: Analyze the data above. Which structural features correlate with HIGH solubility? Which correlate with LOW solubility?

Step 2: Using that analysis, design ONE new molecule predicted to have a HIGHER LogS than any molecule above.

Think carefully and avoid trivial solutions:
- Do NOT simply extend a homologous series or repeat one motif (e.g. adding more -CH(OH)- or -CH2- units to lengthen a chain). That exploits the scoring function but is not a real discovery.
- Propose a structurally distinct, realistic molecule that a practicing chemist would consider genuinely synthesizable and useful.
- Favor chemical realism over blindly maximizing one descriptor.

Requirements:
- Must be a real, synthesizable organic compound
- Do not repeat any molecule already listed
- Respond ONLY with valid JSON, no other text

{
  "smiles": "<valid SMILES string>",
  "name": "<common or IUPAC name>",
  "reasoning": "<reasoning>"
}
```

---

## Variant C: Constrained (+ pragmatic)

Hard constraints box out descriptor exploitation by capping size and ring count; the
pragmatic wording (added after Variant A/B both got exploited) blocks the
homologous-chain and salt-mixture loopholes explicitly.

```
You are a computational chemist optimizing aqueous solubility.

Below are molecules that have been tested, with their LogS values (higher = more soluble in water):

[seed block]

Step 1: Analyze the data above. Which structural features correlate with HIGH solubility? Which correlate with LOW solubility?

Step 2: Using that analysis, design ONE new molecule predicted to have a HIGHER LogS than any molecule above, obeying every constraint below.

Hard constraints:
- Molecular weight must be under 150 Da
- Must contain at least one hydroxyl (OH), amine (NH2), or carbonyl (C=O) group
- No more than one aromatic ring
- Must be a SINGLE, connected molecule: no salts, mixtures, co-crystals, or disconnected components (no '.' in the SMILES)
- Must be a real, synthesizable organic compound
- Do not repeat any molecule already listed

Think carefully and avoid trivial solutions:
- Do NOT simply extend a homologous series or repeat one motif (e.g. adding more -CH(OH)- or -CH2- units). That exploits the scoring function but is not a real discovery.
- Favor chemical realism over blindly maximizing one descriptor.

Respond ONLY with valid JSON, no other text:

{
  "smiles": "<valid SMILES string>",
  "name": "<common or IUPAC name>",
  "reasoning": "<reasoning>"
}
```
