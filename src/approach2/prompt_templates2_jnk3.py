"""
Prompt generation for Approach 2 on the JNK3 property (LLM as regressor).
Same shape as prompt_templates2.py: LLM predicts a score per candidate, does not
design molecules. Field renamed predicted_logs -> predicted_jnk3, range 0-1.
"""

DOMAIN_HINT = (
    "Domain hint: JNK3 is an ATP-competitive kinase. Real inhibitors typically bind "
    "the kinase hinge region through a heteroaromatic core capable of 1-2 hydrogen "
    "bonds (e.g. aminopyridine, aminopyrimidine, indazole, quinazoline, purine-like "
    "scaffolds), often with a solvent-exposed substituent and moderate molecular "
    "weight (roughly 250-500 Da). Simple aliphatic or non-aromatic molecules "
    "essentially never score above 0.\n\n"
)


def _format_observed(observed):
    lines = []
    for i, m in enumerate(observed, 1):
        name = f" ({m['name']})" if m.get("name") else ""
        lines.append(f"{i:2d}. {m['smiles']}{name}: JNK3 = {m['jnk3']:.3f}")
    return "\n".join(lines)


def _format_candidates(candidates):
    lines = []
    for i, c in enumerate(candidates, 1):
        name = f" ({c['name']})" if c.get("name") else ""
        lines.append(f"{i:2d}. {c['smiles']}{name}")
    return "\n".join(lines)


def generate_regression_prompt(observed, candidates):
    """observed: list of {name, smiles, jnk3}. candidates: list of {name, smiles}."""
    return f"""You are a computational chemist acting as a JNK3 kinase-inhibition prediction model.

{DOMAIN_HINT}Below are molecules already measured, with their JNK3 scores (0 to 1, higher = more likely a JNK3 inhibitor):

{_format_observed(observed)}

Now predict the JNK3 score for EACH candidate molecule below. For each one give:
  - predicted_jnk3: your best numeric estimate in [0, 1]
  - confidence: an integer 1-10 (10 = very sure, 1 = wild guess)

Candidates:
{_format_candidates(candidates)}

Respond ONLY with a JSON array as snippet code, one object per candidate, no other text:

[
  {{"id": 1, "smiles": "<candidate SMILES>", "predicted_jnk3": <number>, "confidence": <1-10>}},
  {{"id": 2, "smiles": "<candidate SMILES>", "predicted_jnk3": <number>, "confidence": <1-10>}}
]

Predict every candidate. Keep the same id and smiles shown above.
"""
