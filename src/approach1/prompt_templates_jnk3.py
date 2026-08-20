"""
Prompt generation for Approach 1 on the JNK3 property.
Same 3 variants as prompt_templates.py (A: minimal, B: chain-of-thought,
C: constrained), wording swapped from LogS/solubility to JNK3 inhibition score.
"""

JSON_BLOCK = """{
  "smiles": "<valid SMILES string>",
  "name": "<common or IUPAC name>",
  "reasoning": "<reasoning>"
}"""

HEADER = (
    "You are a computational chemist optimizing predicted JNK3 kinase inhibition.\n\n"
    "Domain hint: JNK3 is an ATP-competitive kinase. Real inhibitors typically bind "
    "the kinase hinge region through a heteroaromatic core capable of 1-2 hydrogen "
    "bonds (e.g. aminopyridine, aminopyrimidine, indazole, quinazoline, purine-like "
    "scaffolds), often with a solvent-exposed substituent and moderate molecular "
    "weight (roughly 250-500 Da). Simple aliphatic or non-aromatic molecules "
    "essentially never score above 0.\n\n"
    "Below are molecules that have been tested, with their JNK3 scores "
    "(0 to 1, higher = more likely a JNK3 inhibitor):\n\n"
)

REQUIREMENTS = (
    "Requirements:\n"
    "- Must be a real, synthesizable organic compound\n"
    "- Do not repeat any molecule already listed\n"
    "- Respond ONLY with valid JSON, no other text\n\n"
)


def _format_known(known):
    rows = sorted(known, key=lambda m: m["jnk3"], reverse=True)
    lines = []
    for i, m in enumerate(rows, 1):
        lines.append(f"{i:>2}. {m['name']:<14} {m['smiles']:<45} JNK3 = {m['jnk3']:.3f}")
    return "\n".join(lines)


def generate_prompt(known, variant="B"):
    """known: list of dicts {name, smiles, jnk3}. variant: "A", "B", or "C"."""
    table = _format_known(known)
    body = HEADER + table + "\n\n"

    if variant == "A":
        instruction = (
            "Step 1: Analyze the data above. Which structural features correlate with "
            "HIGH JNK3 score? Which correlate with LOW JNK3 score?\n\n"
            "Step 2: Using that analysis, design ONE new molecule predicted to have a "
            "HIGHER JNK3 score than any molecule above.\n\n"
        )
        return body + instruction + REQUIREMENTS + JSON_BLOCK

    if variant == "B":
        instruction = (
            "Step 1: Analyze the data above. Which structural features correlate with "
            "HIGH JNK3 score? Which correlate with LOW JNK3 score?\n\n"
            "Step 2: Using that analysis, design ONE new molecule predicted to have a "
            "HIGHER JNK3 score than any molecule above.\n\n"
            "Think carefully and avoid trivial solutions:\n"
            "- Do NOT simply extend a homologous series or repeat one motif. That "
            "exploits the scoring function but is not a real discovery.\n"
            "- Propose a structurally distinct, realistic molecule that a practicing "
            "chemist would consider genuinely synthesizable and useful.\n"
            "- Favor chemical realism over blindly maximizing one descriptor.\n\n"
        )
        return body + instruction + REQUIREMENTS + JSON_BLOCK

    if variant == "C":
        instruction = (
            "Step 1: Analyze the data above. Which structural features correlate with "
            "HIGH JNK3 score? Which correlate with LOW JNK3 score?\n\n"
            "Step 2: Using that analysis, design ONE new molecule predicted to have a "
            "HIGHER JNK3 score than any molecule above, obeying every constraint below.\n\n"
            "Hard constraints:\n"
            "- Molecular weight must be under 500 Da\n"
            "- Must contain at least one aromatic or heteroaromatic ring (common in "
            "kinase-hinge binders)\n"
            "- Must be a SINGLE, connected molecule: no salts, mixtures, co-crystals, "
            "or disconnected components (no '.' in the SMILES)\n"
            "- Must be a real, synthesizable organic compound\n"
            "- Do not repeat any molecule already listed\n\n"
            "Think carefully and avoid trivial solutions:\n"
            "- Do NOT simply extend a homologous series or repeat one motif. That "
            "exploits the scoring function but is not a real discovery.\n"
            "- Favor chemical realism over blindly maximizing one descriptor.\n\n"
            "Respond ONLY with valid JSON, no other text:\n\n"
        )
        return body + instruction + JSON_BLOCK

    raise ValueError(f"Unknown variant: {variant} (expected A, B, or C)")
