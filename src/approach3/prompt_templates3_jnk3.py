"""
Prompt generation for Approach 3 on the JNK3 property (LLM as pairwise ranker).
Same shape as prompt_templates3.py: no numbers, just A/B duel winners.
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


def _format_duels(candidates, pairs):
    lines = []
    for q, (i, j) in enumerate(pairs, 1):
        a, b = candidates[i], candidates[j]
        a_name = f" ({a['name']})" if a.get("name") else ""
        b_name = f" ({b['name']})" if b.get("name") else ""
        lines.append(f"Q{q}: A = {a['smiles']}{a_name}   vs   B = {b['smiles']}{b_name}")
    return "\n".join(lines)


def generate_pairwise_prompt(observed, candidates, pairs):
    """observed: list of {name, smiles, jnk3}. candidates: list of {name, smiles}."""
    return f"""You are a computational chemist comparing molecules by predicted JNK3 kinase inhibition (0 to 1, higher = more likely a JNK3 inhibitor).

{DOMAIN_HINT}For reference, here are molecules already measured with their JNK3 scores:

{_format_observed(observed)}

Below are pairs of molecules. For EACH pair, decide which molecule is MORE likely to inhibit JNK3. Do not estimate numbers, just choose A or B.

{_format_duels(candidates, pairs)}

Respond ONLY with a JSON array in snippet code, one object per question, no other text:

[
  {{"q": 1, "winner": "A"}},
  {{"q": 2, "winner": "B"}}
]

Answer every question. "winner" must be exactly "A" or "B".
"""
