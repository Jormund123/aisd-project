"""
Prompt generation for Approach 2 (LLM as regressor).

Difference from Approach 1: the LLM does NOT design molecules. It is a surrogate
model. We show it the molecules already measured (with their true ESOL LogS) and a
list of candidate molecules, and ask it to PREDICT each candidate's LogS plus a
confidence (1-10). The acquisition function, not the LLM, decides what to test next.
"""


def _format_observed(observed):
    """Numbered 'SMILES (name): LogS' lines for the already-measured molecules."""
    lines = []
    for i, m in enumerate(observed, 1):
        name = f" ({m['name']})" if m.get("name") else ""
        lines.append(f"{i:2d}. {m['smiles']}{name}: LogS = {m['logs']:+.3f}")
    return "\n".join(lines)


def _format_candidates(candidates):
    """Numbered candidate lines. id is the 1-based index the LLM must echo back."""
    lines = []
    for i, c in enumerate(candidates, 1):
        name = f" ({c['name']})" if c.get("name") else ""
        lines.append(f"{i:2d}. {c['smiles']}{name}")
    return "\n".join(lines)


def generate_regression_prompt(observed, candidates):
    """Build the Approach 2 prediction prompt.

    observed:   list of {name, smiles, logs} already measured (the LLM's context).
    candidates: list of {name, smiles} to predict (id = position in this list).
    """
    return f"""You are a computational chemist acting as a solubility prediction model.

Below are molecules already measured, with their LogS values (higher = more soluble in water):

{_format_observed(observed)}

Now predict the LogS for EACH candidate molecule below. For each one give:
  - predicted_logs: your best numeric estimate of its LogS (same scale as above)
  - confidence: an integer 1-10 (10 = very sure, 1 = wild guess)

Candidates:
{_format_candidates(candidates)}

Respond ONLY with a JSON array as snippet code, one object per candidate, no other text:

[
  {{"id": 1, "smiles": "<candidate SMILES>", "predicted_logs": <number>, "confidence": <1-10>}},
  {{"id": 2, "smiles": "<candidate SMILES>", "predicted_logs": <number>, "confidence": <1-10>}}
]

Predict every candidate. Keep the same id and smiles shown above.
"""
