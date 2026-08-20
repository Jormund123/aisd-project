"""
Prompt generation for Approach 3 (LLM as a pairwise ranker).

Difference from Approach 2: the LLM gives NO numbers. For each duel it only names the
more soluble molecule (A or B). We show the seed molecules with their true ESOL LogS
as context (same block as Approach 2), then list this round's duels as questions.
"""


def _format_observed(observed):
    """Numbered 'SMILES (name): LogS' lines for the already-measured molecules."""
    lines = []
    for i, m in enumerate(observed, 1):
        name = f" ({m['name']})" if m.get("name") else ""
        lines.append(f"{i:2d}. {m['smiles']}{name}: LogS = {m['logs']:+.3f}")
    return "\n".join(lines)


def _format_duels(candidates, pairs):
    """One 'Qk: A = ...  vs  B = ...' line per (i, j) pair to be judged."""
    lines = []
    for q, (i, j) in enumerate(pairs, 1):
        a, b = candidates[i], candidates[j]
        a_name = f" ({a['name']})" if a.get("name") else ""
        b_name = f" ({b['name']})" if b.get("name") else ""
        lines.append(f"Q{q}: A = {a['smiles']}{a_name}   vs   B = {b['smiles']}{b_name}")
    return "\n".join(lines)


def generate_pairwise_prompt(observed, candidates, pairs):
    """Build the Approach 3 duel prompt.

    observed:   list of {name, smiles, logs} already measured (context, with truth).
    candidates: list of {name, smiles} (the fixed pool; indices are stable).
    pairs:      list of (i, j) index pairs into candidates, the duels to judge now.
    """
    return f"""You are a computational chemist comparing molecules by aqueous solubility (LogS, higher = more soluble in water).

For reference, here are molecules already measured with their LogS values:

{_format_observed(observed)}

Below are pairs of molecules. For EACH pair, decide which molecule is MORE soluble in water. Do not estimate numbers, just choose A or B.

{_format_duels(candidates, pairs)}

Respond ONLY with a JSON array in snippet code, one object per question, no other text:

[
  {{"q": 1, "winner": "A"}},
  {{"q": 2, "winner": "B"}}
]

Answer every question. "winner" must be exactly "A" or "B".
"""
