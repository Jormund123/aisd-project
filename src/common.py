"""
Everything shared by 2+ of the numbered pipeline stages (_02_approach1.py,
_03_approach2.py, _04_approach3.py, _05_analysis.py, _06_plots.py), in one file so
adding a property or an approach doesn't mean adding more files. Sections:

  1. validate_smiles           -- generic RDKit validation/canonicalization
  2. PROPERTIES registry       -- ESOL (formula) + JNK3 (oracle_env subprocess bridge)
  3. GP                        -- Gaussian Process surrogate (latent-space BO)
  4. SelfiesVAE / VAEModel     -- pretrained molecule encoder/decoder
  5. generate_candidates / rank_candidates / EI / UCB  -- generative BO helpers
  6. Bradley-Terry / PBO / duel utilities              -- Approach 3 ranking math
  7. metrics                   -- pure evaluation-metric functions
  8. manual-LLM paste parsing  -- read_pasted_json / extract_json / extract_json_array

Run directly as a CLI to validate+score one SMILES (replaces evaluate.py):
    python common.py --property jnk3 "CCO"
"""

import atexit
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy.stats import norm, spearmanr, kendalltau

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# =================================================================================
# 1. Generic SMILES validation
# =================================================================================

def validate_smiles(smiles):
    """Returns (mol, canonical_smiles) or (None, None) if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    return mol, Chem.MolToSmiles(mol)


# =================================================================================
# 2. Property registry: ESOL (formula) + JNK3 (subprocess-bridged oracle)
# =================================================================================

# -- ESOL: Delaney's method, adapted from PatWalters/solubility --------------------
_ESOL_INTERCEPT = 0.16
_ESOL_COEF = {"logp": -0.63, "mw": -0.0062, "rb": 0.066, "ap": -0.74}


def _esol_aromatic_proportion(mol):
    aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    heavy = mol.GetNumHeavyAtoms()
    return aromatic / heavy if heavy else 0.0


def _score_esol(smiles):
    """Validate + compute ESOL LogS. Returns float or None if invalid."""
    mol, _ = validate_smiles(smiles)
    if mol is None:
        return None
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    ap = _esol_aromatic_proportion(mol)
    logs = (_ESOL_INTERCEPT + _ESOL_COEF["logp"] * logp + _ESOL_COEF["mw"] * mw
            + _ESOL_COEF["rb"] * rb + _ESOL_COEF["ap"] * ap)
    return round(logs, 4)


def _score_esol_batch(smiles_list):
    return [_score_esol(s) for s in smiles_list]


def _esol_baseline_pool():
    """AqSolDB SMILES scored with ESOL, for the random-baseline comparison."""
    df = pd.read_csv(os.path.join(DATA_DIR, "aqsoldb.csv"))
    scores = np.array([s for s in _score_esol_batch(df["SMILES"]) if s is not None])
    return scores


# -- JNK3: TDC Oracle('JNK3'), bridged via a subprocess into oracle_env -----------
# The pretrained JNK3 classifier is a scikit-learn 0.23 (2020) pickle; modern
# sklearn (>=1.3) changed its tree node format and can't load it, and no sklearn
# old enough to load it ships a wheel for this repo's Python. So JNK3 scoring runs
# in oracle_env (see scripts/setup_jnk3_oracle_env.sh), invoked as a persistent
# subprocess -- see docs/jnk3_implementation_plan.md for the full story.

_JNK3_ORACLE_PYTHON = os.path.join(os.path.dirname(__file__), "..", "oracle_env", "bin", "python3")
_JNK3_WORKER = os.path.join(os.path.dirname(__file__), "jnk3_worker.py")
_jnk3_proc = None


def _jnk3_worker_process():
    global _jnk3_proc
    if _jnk3_proc is None or _jnk3_proc.poll() is not None:
        if not os.path.exists(_JNK3_ORACLE_PYTHON):
            raise RuntimeError(
                "oracle_env not found. Run scripts/setup_jnk3_oracle_env.sh first.")
        _jnk3_proc = subprocess.Popen(
            [_JNK3_ORACLE_PYTHON, _JNK3_WORKER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        atexit.register(_jnk3_proc.terminate)
    return _jnk3_proc


def _score_jnk3(smiles):
    mol, canonical = validate_smiles(smiles)
    if mol is None:
        return None
    proc = _jnk3_worker_process()
    proc.stdin.write(canonical + "\n")
    proc.stdin.flush()
    raw = proc.stdout.readline().strip()
    return round(float(raw), 4) if raw else None


def _score_jnk3_batch(smiles_list):
    canon = [validate_smiles(s)[1] for s in smiles_list]
    proc = _jnk3_worker_process()
    pending = [c for c in canon if c is not None]
    for c in pending:
        proc.stdin.write(c + "\n")
    proc.stdin.flush()
    scores = {}
    for c in pending:
        raw = proc.stdout.readline().strip()
        if c not in scores:
            scores[c] = round(float(raw), 4) if raw else None
    return [scores.get(c) if c is not None else None for c in canon]


def _jnk3_baseline_pool():
    """Pre-scored ZINC sample (data/jnk3_scored_pool.csv) for the random baseline.
    Reused rather than rescored: scoring 3000 molecules through the oracle_env
    subprocess is slow, and this pool is already committed from setup."""
    df = pd.read_csv(os.path.join(DATA_DIR, "jnk3_scored_pool.csv"))
    return df["jnk3_score"].to_numpy(dtype=float)


_JNK3_DOMAIN_HINT = (
    "Domain hint: JNK3 is an ATP-competitive kinase. Real inhibitors typically bind "
    "the kinase hinge region through a heteroaromatic core capable of 1-2 hydrogen "
    "bonds (e.g. aminopyridine, aminopyrimidine, indazole, quinazoline, purine-like "
    "scaffolds), often with a solvent-exposed substituent and moderate molecular "
    "weight (roughly 250-500 Da). Simple aliphatic or non-aromatic molecules "
    "essentially never score above 0.\n\n"
)

PROPERTIES = {
    "esol": {
        "label": "LogS (ESOL)",
        "score": _score_esol,
        "score_batch": _score_esol_batch,
        "baseline_pool": _esol_baseline_pool,
        "domain_hint": "",
        "prompt_property_desc": "aqueous solubility (LogS, higher = more soluble in water)",
        "variant_c_constraints": (
            "- Molecular weight must be under 150 Da\n"
            "- Must contain at least one hydroxyl (OH), amine (NH2), or carbonyl (C=O) group\n"
            "- No more than one aromatic ring\n"
        ),
    },
    "jnk3": {
        "label": "JNK3 score",
        "score": _score_jnk3,
        "score_batch": _score_jnk3_batch,
        "baseline_pool": _jnk3_baseline_pool,
        "domain_hint": _JNK3_DOMAIN_HINT,
        "prompt_property_desc": "JNK3 kinase inhibition (0 to 1, higher = more likely a JNK3 inhibitor)",
        "variant_c_constraints": (
            "- Molecular weight must be under 500 Da\n"
            "- Must contain a heteroaromatic hinge-binding scaffold (e.g. aminopyridine, "
            "aminopyrimidine, indazole, quinazoline, or purine-like core)\n"
            "- Must have at least one hydrogen-bond donor (NH or OH) positioned to reach "
            "the kinase hinge\n"
            "- No more than one solvent-exposed aromatic substituent (avoid over-decorating "
            "the core with extra rings)\n"
        ),
    },
}


def run_baseline(property_name, n=15, k=1000, seed=0):
    """K trials of best-of-N random draws from the property's baseline pool.
    Returns (mean, std, ceiling) -- ceiling is the pool's best score."""
    pool = PROPERTIES[property_name]["baseline_pool"]()
    rng = np.random.default_rng(seed)
    bests = np.empty(k)
    for i in range(k):
        bests[i] = rng.choice(pool, size=min(n, len(pool)), replace=False).max()
    return float(bests.mean()), float(bests.std()), float(pool.max())


# =================================================================================
# 3. GP: RBF-kernel Gaussian Process (latent-space BO surrogate)
# =================================================================================

def _rbf(A, B, lengthscale, signal_var):
    a2 = np.sum(A * A, axis=1)[:, None]
    b2 = np.sum(B * B, axis=1)[None, :]
    d2 = np.clip(a2 + b2 - 2.0 * A @ B.T, 0.0, None)
    return signal_var * np.exp(-0.5 * d2 / (lengthscale ** 2))


class GP:
    """RBF-kernel GP with per-dimension input standardization and standardized
    targets. Usage: gp = GP().fit(Z, y); mu, sigma = gp.posterior(Zstar)."""

    def __init__(self, noise=1e-2, signal_var=1.0,
                 lengthscales=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0)):
        self.noise = float(noise)
        self.signal_var = float(signal_var)
        self.lengthscale_grid = tuple(lengthscales)
        self._fitted = False

    def _log_marginal_likelihood(self, K, y):
        n = len(y)
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        lml = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
        return lml, L, alpha

    def fit(self, Z, y):
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if Z.ndim == 1:
            Z = Z[:, None]
        n = len(y)

        self.z_mean = Z.mean(axis=0)
        self.z_std = Z.std(axis=0)
        self.z_std[self.z_std < 1e-8] = 1.0
        Zs = (Z - self.z_mean) / self.z_std

        self.y_mean = float(y.mean())
        self.y_std = float(y.std())
        if self.y_std < 1e-8:
            self.y_std = 1.0
        ys = (y - self.y_mean) / self.y_std

        best = None
        for ls in self.lengthscale_grid:
            K = _rbf(Zs, Zs, ls, self.signal_var) + self.noise * np.eye(n)
            lml, L, alpha = self._log_marginal_likelihood(K, ys)
            if best is None or lml > best[0]:
                best = (lml, ls, L, alpha)

        _, self.lengthscale, self._L, self._alpha = best
        self._Zs = Zs
        self._fitted = True
        return self

    def posterior(self, Zstar):
        if not self._fitted:
            raise RuntimeError("GP.posterior called before fit")
        Zstar = np.asarray(Zstar, dtype=float)
        if Zstar.ndim == 1:
            Zstar = Zstar[:, None]
        Zs = (Zstar - self.z_mean) / self.z_std

        Ks = _rbf(Zs, self._Zs, self.lengthscale, self.signal_var)
        mu_s = Ks @ self._alpha

        v = np.linalg.solve(self._L, Ks.T)
        var_s = self.signal_var - np.sum(v * v, axis=0)
        var_s = np.clip(var_s, 0.0, None)

        mu = mu_s * self.y_std + self.y_mean
        sigma = np.sqrt(var_s) * self.y_std
        return mu, sigma


# =================================================================================
# 4. SelfiesVAE: pretrained molecule encoder/decoder
# =================================================================================

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIALS = [PAD, BOS, EOS]


def smiles_to_selfies(smiles):
    try:
        return sf.encoder(smiles)
    except Exception:  # noqa: BLE001 - selfies raises several encoder errors
        return None


def build_vocab(selfies_list):
    alphabet = set()
    for s in selfies_list:
        if s is None:
            continue
        alphabet.update(sf.split_selfies(s))
    tokens = SPECIALS + sorted(alphabet)
    stoi = {t: i for i, t in enumerate(tokens)}
    return tokens, stoi


def selfies_to_ids(s, stoi, max_len):
    toks = list(sf.split_selfies(s))[: max_len - 2]
    ids = [stoi[BOS]] + [stoi[t] for t in toks if t in stoi] + [stoi[EOS]]
    ids = ids[:max_len]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


class VAEModel(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=256, latent=64, pad_idx=0):
        super().__init__()
        self.latent = latent
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, emb, padding_idx=pad_idx)
        self.encoder = nn.GRU(emb, hidden, batch_first=True)
        self.h2mu = nn.Linear(hidden, latent)
        self.h2logvar = nn.Linear(hidden, latent)
        self.z2h = nn.Linear(latent, hidden)
        self.decoder = nn.GRU(emb + latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def encode(self, x):
        e = self.embed(x)
        lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)
        h = h.squeeze(0)
        return self.h2mu(h), self.h2logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode_train(self, dec_input, z):
        e = self.embed(dec_input)
        zt = z.unsqueeze(1).expand(-1, e.size(1), -1)
        h0 = torch.tanh(self.z2h(z)).unsqueeze(0)
        out, _ = self.decoder(torch.cat([e, zt], dim=-1), h0)
        return self.out(out)

    def forward(self, x, dec_input=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        di = x[:, :-1] if dec_input is None else dec_input
        return self.decode_train(di, z), mu, logvar

    @torch.no_grad()
    def generate(self, z, max_len, bos_idx, eos_idx):
        h = torch.tanh(self.z2h(z)).unsqueeze(0)
        tok = torch.full((z.shape[0], 1), bos_idx, dtype=torch.long, device=z.device)
        zt = z.unsqueeze(1)
        outs = []
        for _ in range(max_len):
            e = torch.cat([self.embed(tok), zt], dim=-1)
            o, h = self.decoder(e, h)
            nxt = self.out(o[:, -1]).argmax(-1, keepdim=True)
            outs.append(nxt)
            tok = nxt
        return torch.cat(outs, dim=1)


class SelfiesVAE:
    """Owns vocabulary + a trained VAEModel; exposes the BO-facing API."""

    def __init__(self, model, tokens, config, device="cpu"):
        self.model = model.to(device).eval()
        self.tokens = tokens
        self.stoi = {t: i for i, t in enumerate(tokens)}
        self.config = config
        self.device = device
        self.max_len = config["max_len"]

    @property
    def latent_dim(self):
        return self.config["latent"]

    @torch.no_grad()
    def encode(self, smiles_list):
        ids, kept = [], []
        for smi in smiles_list:
            s = smiles_to_selfies(smi)
            if s is None:
                continue
            ids.append(selfies_to_ids(s, self.stoi, self.max_len))
            kept.append(smi)
        if not ids:
            return np.zeros((0, self.latent_dim)), []
        x = torch.tensor(ids, dtype=torch.long, device=self.device)
        mu, _ = self.model.encode(x)
        return mu.cpu().numpy(), kept

    @torch.no_grad()
    def decode(self, Z):
        Z = np.asarray(Z, dtype=float)
        if Z.ndim == 1:
            Z = Z[None, :]
        z = torch.tensor(Z, dtype=torch.float32, device=self.device)
        id_batch = self.model.generate(z, self.max_len, self.stoi[BOS], self.stoi[EOS])
        results = []
        for row in id_batch.cpu().numpy():
            toks = []
            for i in row:
                t = self.tokens[i]
                if t == EOS:
                    break
                if t in (PAD, BOS):
                    continue
                toks.append(t)
            smi = None
            if toks:
                try:
                    smi = sf.decoder("".join(toks))
                except Exception:  # noqa: BLE001
                    smi = None
            if smi:
                _, canonical = validate_smiles(smi)
                smi = canonical
            results.append(smi)
        return results

    def sample_latents(self, n, anchors=None, scale=1.0, jitter=0.5, rng=None):
        rng = rng or np.random.default_rng()
        d = self.latent_dim
        if anchors is None or len(anchors) == 0:
            return rng.normal(0.0, scale, size=(n, d))
        anchors = np.asarray(anchors, dtype=float)
        near_n = int(round(0.75 * n))
        prior = rng.normal(0.0, scale, size=(n - near_n, d))
        idx = rng.integers(0, len(anchors), size=near_n)
        near = anchors[idx] + rng.normal(0.0, jitter, size=(near_n, d))
        return np.vstack([near, prior])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(),
                    "tokens": self.tokens, "config": self.config}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = VAEModel(len(ckpt["tokens"]), emb=cfg["emb"], hidden=cfg["hidden"],
                         latent=cfg["latent"], pad_idx=0)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["tokens"], cfg, device=device)


# =================================================================================
# 5. Generative BO helpers: candidate generation + EI/UCB acquisition
# =================================================================================

def reasonable_molecule(smiles, min_heavy=3, max_heavy=40, max_mw=600.0):
    """Design-space sanity filter: keep drug-like sizes, reject decoder pathologies."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    heavy = mol.GetNumHeavyAtoms()
    if heavy < min_heavy or heavy > max_heavy:
        return False
    return Descriptors.MolWt(mol) <= max_mw


def generate_candidates(vae, anchor_Z, n_target, exclude_smiles, rng,
                        oversample=8, max_tries=10, scale=1.0, jitter=0.5,
                        accept_fn=reasonable_molecule):
    """Decode fresh, valid, unique molecules from the VAE latent space. Returns
    (smiles_list, Z) with up to n_target molecules."""
    seen = set(exclude_smiles)
    picked_smiles, picked_Z = [], []
    for _ in range(max_tries):
        if len(picked_smiles) >= n_target:
            break
        Z = vae.sample_latents(n_target * oversample, anchors=anchor_Z,
                               scale=scale, jitter=jitter, rng=rng)
        decoded = vae.decode(Z)
        fresh = []
        for smi in decoded:
            if smi and smi not in seen and (accept_fn is None or accept_fn(smi)):
                seen.add(smi)
                fresh.append(smi)
        if not fresh:
            continue
        Zf, kept = vae.encode(fresh)
        for smi, z in zip(kept, Zf):
            picked_smiles.append(smi)
            picked_Z.append(z)
            if len(picked_smiles) >= n_target:
                break
    Z = np.array(picked_Z) if picked_Z else np.zeros((0, vae.latent_dim))
    return picked_smiles, Z


def expected_improvement(mu, sigma, f_best, xi=0.0):
    """EI for maximization. Where sigma <= 0, collapses to max(mu-f_best-xi, 0)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    improvement = mu - f_best - xi
    ei = np.zeros_like(mu)
    pos = sigma > 0
    z = np.zeros_like(mu)
    z[pos] = improvement[pos] / sigma[pos]
    ei[pos] = improvement[pos] * norm.cdf(z[pos]) + sigma[pos] * norm.pdf(z[pos])
    ei[~pos] = np.maximum(improvement[~pos], 0.0)
    return ei


def upper_confidence_bound(mu, sigma, kappa=2.0):
    """UCB for maximization: mu + kappa * sigma."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return mu + kappa * sigma


def rank_candidates(mu, sigma, f_best, acq="ei", kappa=2.0, xi=0.0):
    """Acquisition scores for candidates from GP posterior (mu, sigma). Returns
    (order, scores) where order indexes candidates best-first."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if acq == "ei":
        scores = expected_improvement(mu, sigma, f_best, xi=xi)
    elif acq == "ucb":
        scores = upper_confidence_bound(mu, sigma, kappa=kappa)
    else:
        raise ValueError(f"unknown acquisition '{acq}' (use 'ei' or 'ucb')")
    order = np.argsort(-scores)
    return order, scores


# =================================================================================
# 6. Bradley-Terry / PBO / duel utilities (Approach 3 ranking math)
# =================================================================================

def _split_winner_loser(duels):
    win, lose = [], []
    for d in duels:
        i, j, w = d
        if w != i and w != j:
            raise ValueError(f"duel winner {w} is neither {i} nor {j}")
        loser = j if w == i else i
        win.append(int(w))
        lose.append(int(loser))
    return np.asarray(win, dtype=int), np.asarray(lose, dtype=int)


def win_count_scores(duels, n):
    scores = np.zeros(n, dtype=float)
    if not duels:
        return scores
    win, _ = _split_winner_loser(duels)
    for w in win:
        scores[w] += 1.0
    return scores


def bradley_terry(duels, n, reg=1e-3, max_iter=100, tol=1e-8):
    """Fit Bradley-Terry latent utilities by regularized MLE (Newton's method).
    Returns (utility, std_err), both length-n arrays."""
    u = np.zeros(n, dtype=float)
    if not duels:
        return u, np.full(n, 1.0 / np.sqrt(reg))

    win, lose = _split_winner_loser(duels)

    for _ in range(max_iter):
        d = u[win] - u[lose]
        p = 1.0 / (1.0 + np.exp(-d))
        g = p * (1.0 - p)

        grad = -reg * u
        np.add.at(grad, win, 1.0 - p)
        np.add.at(grad, lose, -(1.0 - p))

        H = -reg * np.eye(n)
        np.add.at(H, (win, win), -g)
        np.add.at(H, (lose, lose), -g)
        np.add.at(H, (win, lose), g)
        np.add.at(H, (lose, win), g)

        step = np.linalg.solve(H, grad)
        u = u - step
        if np.max(np.abs(step)) < tol:
            break

    u = u - u.mean()

    d = u[win] - u[lose]
    p = 1.0 / (1.0 + np.exp(-d))
    g = p * (1.0 - p)
    info = reg * np.eye(n)
    np.add.at(info, (win, win), g)
    np.add.at(info, (lose, lose), g)
    np.add.at(info, (win, lose), -g)
    np.add.at(info, (lose, win), -g)
    cov = np.linalg.inv(info)
    std_err = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return u, std_err


def pbo_acquisition(utility, std_err, evaluated_mask, kappa=2.0):
    """argmax over UNEVALUATED of utility + kappa*std_err. None if all evaluated."""
    utility = np.asarray(utility, dtype=float)
    std_err = np.asarray(std_err, dtype=float)
    mask = np.asarray(evaluated_mask, dtype=bool)
    if mask.all():
        return None
    scores = utility + kappa * std_err
    scores = np.where(mask, -np.inf, scores)
    return int(np.argmax(scores))


def pairwise_accuracy(pred, true):
    """Fraction of pairs ordered the same way by pred and by true.

    Only pairs tied in TRUE are skipped -- a predicted tie where true differs counts
    as a miss (sign(0) != sign(nonzero)), not an excluded pair. This is the semantics
    the published report numbers were generated with (was analysis_common.py's
    version; a second, slightly different implementation also existed in the old
    per-approach-3 pairwise.py module -- skipping pairs tied in EITHER pred or true --
    which very rarely disagrees since predicted ties are almost never exact floats,
    but did on real LLM output. Consolidated on this one, the report-defining one.)
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    n = len(pred)
    correct = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if true[i] == true[j]:
                continue
            total += 1
            if np.sign(pred[i] - pred[j]) == np.sign(true[i] - true[j]):
                correct += 1
    return correct / total if total else float("nan")


def ranking_metrics(pred_utility, true_values):
    """{spearman, kendall, pairwise_accuracy} of predicted vs true ranking."""
    pred = np.asarray(pred_utility, dtype=float)
    true = np.asarray(true_values, dtype=float)
    if len(pred) < 2:
        return {"spearman": float("nan"), "kendall": float("nan"),
                "pairwise_accuracy": float("nan")}
    rho = float(spearmanr(pred, true).statistic)
    tau = float(kendalltau(pred, true).statistic)
    return {"spearman": rho, "kendall": tau,
            "pairwise_accuracy": pairwise_accuracy(pred, true)}


def top_k_so_far_by_round(rounds, true_vals, k=3):
    """Mean true score of the k best molecules found up to and including each round.
    Returns (round_numbers, mean_topk)."""
    rounds = np.asarray(rounds, int)
    true_vals = np.asarray(true_vals, float)
    out_rounds, out_vals = [], []
    for r in sorted(set(rounds.tolist())):
        seen = true_vals[rounds <= r]
        topk = np.sort(seen)[::-1][:k]
        out_rounds.append(r)
        out_vals.append(float(np.mean(topk)))
    return out_rounds, out_vals


def oracle_answers(pairs, pool_true, noise, rng):
    """--auto 'LLM': answer each duel by true value, flipping with prob `noise`."""
    duels = []
    for (i, j) in pairs:
        win = i if pool_true[i] >= pool_true[j] else j
        if noise > 0 and rng.random() < noise:
            win = j if win == i else i
        duels.append((i, j, int(win)))
    return duels


def parse_pairwise(text, pairs):
    """Map a pasted JSON array of {q, winner} answers back onto duels. Returns
    (i, j, winner) list, or None if nothing usable was parsed."""
    arr = extract_json_array(text)
    if not arr:
        return None
    duels = []
    for obj in arr:
        if not isinstance(obj, dict):
            continue
        try:
            q = int(obj.get("q"))
        except (TypeError, ValueError):
            continue
        if not (1 <= q <= len(pairs)):
            continue
        w = str(obj.get("winner", "")).strip().upper()
        i, j = pairs[q - 1]
        if w == "A":
            duels.append((i, j, i))
        elif w == "B":
            duels.append((i, j, j))
    return duels


# =================================================================================
# 7. Metrics: pure evaluation-metric functions
# =================================================================================

def best_found(best_so_far, start):
    traj = np.asarray(best_so_far, dtype=float)
    return float(max(traj.max(), start)) if traj.size else float(start)


def simple_regret(best_found_value, ceiling):
    if ceiling is None:
        return float("nan")
    return float(ceiling - best_found_value)


def normalized_score(best_found_value, start, ceiling):
    if ceiling is None or ceiling - start <= 1e-9:
        return float("nan")
    return float((best_found_value - start) / (ceiling - start))


def improvement_over_seed(best_found_value, start):
    return float(best_found_value - start)


def first_improvement_iter(best_so_far, start, iterations=None):
    traj = np.asarray(best_so_far, dtype=float)
    iters = list(range(1, len(traj) + 1)) if iterations is None else list(iterations)
    for it, v in zip(iters, traj):
        if v > start + 1e-12:
            return int(it)
    return None


def iters_to_ceiling(best_so_far, ceiling, iterations=None, tol=1e-6):
    if ceiling is None:
        return None
    traj = np.asarray(best_so_far, dtype=float)
    iters = list(range(1, len(traj) + 1)) if iterations is None else list(iterations)
    for it, v in zip(iters, traj):
        if v >= ceiling - tol:
            return int(it)
    return None


def success_rate(best_so_far, start):
    traj = np.asarray(best_so_far, dtype=float)
    if traj.size == 0:
        return float("nan")
    prev = np.concatenate(([start], traj[:-1]))
    return float(np.mean(traj > prev + 1e-12))


def regression_metrics(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    n = len(pred)
    if n == 0:
        return {k: float("nan") for k in ("mae", "rmse", "bias", "pearson", "r2")}
    err = pred - true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    pearson = float(np.corrcoef(pred, true)[0, 1]) if n > 1 else float("nan")
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "bias": bias, "pearson": pearson, "r2": r2}


def sign_accuracy(pred, true, threshold):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    sp = np.sign(pred - threshold)
    st = np.sign(true - threshold)
    keep = (sp != 0) & (st != 0)
    if not np.any(keep):
        return float("nan")
    return float(np.mean(sp[keep] == st[keep]))


def top1_hit(pred_utility, true):
    pred = np.asarray(pred_utility, dtype=float)
    true = np.asarray(true, dtype=float)
    if len(pred) == 0:
        return float("nan")
    return 1.0 if int(np.argmax(pred)) == int(np.argmax(true)) else 0.0


def ndcg_at_k(pred_utility, true, k=5):
    pred = np.asarray(pred_utility, dtype=float)
    true = np.asarray(true, dtype=float)
    n = len(pred)
    if n == 0:
        return float("nan")
    k = min(k, n)
    rel = true - true.min()
    if rel.max() <= 0:
        return float("nan")
    pred_order = np.argsort(-pred, kind="stable")[:k]
    ideal_order = np.argsort(-rel, kind="stable")[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(rel[pred_order] * discounts))
    idcg = float(np.sum(rel[ideal_order] * discounts))
    return dcg / idcg if idcg > 0 else float("nan")


def hit_rate(values, start):
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return float("nan")
    return float(np.mean(vals > start + 1e-12))


def regression_error(pred, true):
    """MAE, RMSE, bias (mean pred - true) for a regressor's predictions."""
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    err = pred - true
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err))}


# =================================================================================
# 8. Manual-LLM paste parsing (copy-paste workflow: no API access)
# =================================================================================

def read_pasted_json(prompt="\nPaste LLM response, then press Enter on an empty line:"):
    print(prompt)
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "":
            if lines:
                break
            continue
        lines.append(line)
    return "\n".join(lines)


def extract_json(text):
    """Pull the first {...} object out of pasted text and parse it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def extract_json_array(text):
    """Pull the first [...] array out of pasted text and parse it to a list."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def match_predictions(preds, candidates):
    """Align LLM predictions (field 'predicted_score') to candidates by id then
    SMILES. Returns (idx_in_candidates, mu, confidence) parallel lists."""
    by_smiles = {c["smiles"]: j for j, c in enumerate(candidates)}
    idxs, mus, confs = [], [], []
    used = set()
    for p in preds:
        if not isinstance(p, dict) or "predicted_score" not in p:
            continue
        j = None
        pid = p.get("id")
        if isinstance(pid, int) and 1 <= pid <= len(candidates):
            j = pid - 1
        if j is None and "smiles" in p:
            _, canon = validate_smiles(str(p["smiles"]))
            if canon in by_smiles:
                j = by_smiles[canon]
        if j is None or j in used:
            continue
        try:
            mu = float(p["predicted_score"])
            conf = float(p.get("confidence", 5))
        except (TypeError, ValueError):
            continue
        used.add(j)
        idxs.append(j)
        mus.append(mu)
        confs.append(conf)
    return idxs, mus, confs


# =================================================================================
# CLI: python common.py --property jnk3 "SMILES"  (replaces evaluate.py)
# =================================================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Validate + score one SMILES.")
    p.add_argument("--property", default="esol", choices=list(PROPERTIES))
    p.add_argument("smiles")
    args = p.parse_args()

    mol, canonical = validate_smiles(args.smiles)
    if mol is None:
        print(f"INVALID SMILES: {args.smiles}")
        sys.exit(1)
    score = PROPERTIES[args.property]["score"](args.smiles)
    print(f"Input:      {args.smiles}")
    print(f"Canonical:  {canonical}")
    print(f"{PROPERTIES[args.property]['label']}: {score}")
