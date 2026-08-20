"""
Prompt generation for Approach 1 (LLM as Direct Optimizer).
Three variants (A: minimal, B: chain-of-thought, C: constrained).
All LogS values shown are ESOL (consistent scoreboard, see Phase 2 decision).
"""

JSON_BLOCK = """{
  "smiles": "<valid SMILES string>",
  "name": "<common or IUPAC name>",
  "reasoning": "<reasoning>"
}"""

HEADER = (
    "You are a computational chemist optimizing aqueous solubility.\n\n"
    "Below are molecules that have been tested, with their LogS values "
    "(higher = more soluble in water):\n\n"
)

REQUIREMENTS = (
    "Requirements:\n"
    "- Must be a real, synthesizable organic compound\n"
    "- Do not repeat any molecule already listed\n"
    "- Respond ONLY with valid JSON, no other text\n\n"
)


def _format_known(known):
    """Render the tested-molecule list, sorted by LogS descending."""
    rows = sorted(known, key=lambda m: m["logs"], reverse=True)
    lines = []
    for i, m in enumerate(rows, 1):
        sign = "+" if m["logs"] >= 0 else ""
        lines.append(f"{i:>2}. {m['name']:<14} {m['smiles']:<33} LogS = {sign}{m['logs']:.3f}")
    return "\n".join(lines)


def generate_prompt(known, variant="B"):
    """
    Build the prompt for one iteration.
    known: list of dicts {name, smiles, logs} (all molecules tested so far).
    variant: "A", "B", or "C".
    """
    table = _format_known(known)
    body = HEADER + table + "\n\n"

    if variant == "A":
        # Chain-of-thought: analyze first, then design.
        instruction = (
            "Step 1: Analyze the data above. Which structural features correlate with "
            "HIGH solubility? Which correlate with LOW solubility?\n\n"
            "Step 2: Using that analysis, design ONE new molecule predicted to have a "
            "HIGHER LogS than any molecule above.\n\n"
        )
        return body + instruction + REQUIREMENTS + JSON_BLOCK

    if variant == "B":
        # Pragmatic: same reasoning, but forbid trivial homologous-series exploitation.
        instruction = (
            "Step 1: Analyze the data above. Which structural features correlate with "
            "HIGH solubility? Which correlate with LOW solubility?\n\n"
            "Step 2: Using that analysis, design ONE new molecule predicted to have a "
            "HIGHER LogS than any molecule above.\n\n"
            "Think carefully and avoid trivial solutions:\n"
            "- Do NOT simply extend a homologous series or repeat one motif (e.g. adding "
            "more -CH(OH)- or -CH2- units to lengthen a chain). That exploits the scoring "
            "function but is not a real discovery.\n"
            "- Propose a structurally distinct, realistic molecule that a practicing "
            "chemist would consider genuinely synthesizable and useful.\n"
            "- Favor chemical realism over blindly maximizing one descriptor.\n\n"
        )
        return body + instruction + REQUIREMENTS + JSON_BLOCK

    if variant == "C":
        # Constrained + pragmatic. Hard limits box out descriptor exploitation;
        # the anti-trivial wording blocks the homologous-chain and salt-mixture loopholes.
        instruction = (
            "Step 1: Analyze the data above. Which structural features correlate with "
            "HIGH solubility? Which correlate with LOW solubility?\n\n"
            "Step 2: Using that analysis, design ONE new molecule predicted to have a "
            "HIGHER LogS than any molecule above, obeying every constraint below.\n\n"
            "Hard constraints:\n"
            "- Molecular weight must be under 150 Da\n"
            "- Must contain at least one hydroxyl (OH), amine (NH2), or carbonyl (C=O) group\n"
            "- No more than one aromatic ring\n"
            "- Must be a SINGLE, connected molecule: no salts, mixtures, co-crystals, "
            "or disconnected components (no '.' in the SMILES)\n"
            "- Must be a real, synthesizable organic compound\n"
            "- Do not repeat any molecule already listed\n\n"
            "Think carefully and avoid trivial solutions:\n"
            "- Do NOT simply extend a homologous series or repeat one motif (e.g. adding "
            "more -CH(OH)- or -CH2- units). That exploits the scoring function but is not a "
            "real discovery.\n"
            "- Favor chemical realism over blindly maximizing one descriptor.\n\n"
            "Respond ONLY with valid JSON, no other text:\n\n"
        )
        return body + instruction + JSON_BLOCK

    raise ValueError(f"Unknown variant: {variant} (expected A, B, or C)")
