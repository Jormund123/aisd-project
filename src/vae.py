"""
Small SELFIES Variational Autoencoder: the pretrained encoder-decoder that gives the
generative latent-space BO in Approaches 2 and 3 a continuous, decodable search space.

A molecule is written as SELFIES (every SELFIES string decodes to a VALID molecule, so
the decoder can never emit a syntactically broken graph), tokenized, and passed through
a GRU encoder to a latent vector z. A GRU decoder maps z back to a SELFIES string, which
RDKit canonicalizes. Bayesian optimization then searches the z-space and decodes the
points it wants to try into brand-new molecules.

This module has two parts:
  - VAEModel: the torch nn.Module (encoder + decoder + reparameterization).
  - SelfiesVAE: an inference wrapper that owns the vocabulary and exposes
    encode() / decode() / sample_latents() / save() / load() for the BO loop.

Training lives in scripts/train_vae.py, which builds a VAEModel, fits it on AqSolDB, and
saves a checkpoint that SelfiesVAE.load() reads back.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import selfies as sf

sys.path.insert(0, os.path.dirname(__file__))
from esol import validate_smiles  # noqa: E402

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIALS = [PAD, BOS, EOS]


# -- vocabulary + tokenization ---------------------------------------------------
def smiles_to_selfies(smiles):
    """SMILES -> SELFIES string, or None if it cannot be encoded."""
    try:
        return sf.encoder(smiles)
    except Exception:  # noqa: BLE001 - selfies raises several encoder errors
        return None


def build_vocab(selfies_list):
    """Collect the SELFIES alphabet and return (tokens, stoi) with specials first."""
    alphabet = set()
    for s in selfies_list:
        if s is None:
            continue
        alphabet.update(sf.split_selfies(s))
    tokens = SPECIALS + sorted(alphabet)
    stoi = {t: i for i, t in enumerate(tokens)}
    return tokens, stoi


def selfies_to_ids(s, stoi, max_len):
    """SELFIES string -> padded id list [BOS, ..., EOS, PAD...] of length max_len."""
    toks = list(sf.split_selfies(s))[: max_len - 2]
    ids = [stoi[BOS]] + [stoi[t] for t in toks if t in stoi] + [stoi[EOS]]
    ids = ids[:max_len]
    ids += [stoi[PAD]] * (max_len - len(ids))
    return ids


# -- model -----------------------------------------------------------------------
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
        # z is concatenated to the input embedding at EVERY decoder step so the decoder
        # cannot ignore the latent and fall back to an unconditional token model (the
        # failure mode where every latent decodes to the same alkane chain).
        self.decoder = nn.GRU(emb + latent, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def encode(self, x):
        # Pack to true lengths so the encoder's final hidden state reflects the molecule,
        # not the ~60 trailing PAD steps (which otherwise wash the signal out -> the
        # encoder collapses to a constant latent and free-running decode ignores z).
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
        """Teacher-forced decode. dec_input: (B, T) ids; z: (B, latent) -> logits."""
        e = self.embed(dec_input)                                  # (B, T, emb)
        zt = z.unsqueeze(1).expand(-1, e.size(1), -1)              # (B, T, latent)
        h0 = torch.tanh(self.z2h(z)).unsqueeze(0)
        out, _ = self.decoder(torch.cat([e, zt], dim=-1), h0)
        return self.out(out)

    def forward(self, x, dec_input=None):
        """Teacher-forced pass. dec_input overrides x[:,:-1] (for word dropout)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        di = x[:, :-1] if dec_input is None else dec_input
        return self.decode_train(di, z), mu, logvar

    @torch.no_grad()
    def generate(self, z, max_len, bos_idx, eos_idx):
        """Greedy-decode a batch of latents z (B, latent) -> id tensor (B, max_len)."""
        h = torch.tanh(self.z2h(z)).unsqueeze(0)
        tok = torch.full((z.shape[0], 1), bos_idx, dtype=torch.long, device=z.device)
        zt = z.unsqueeze(1)                                        # (B, 1, latent)
        outs = []
        for _ in range(max_len):
            e = torch.cat([self.embed(tok), zt], dim=-1)
            o, h = self.decoder(e, h)
            nxt = self.out(o[:, -1]).argmax(-1, keepdim=True)
            outs.append(nxt)
            tok = nxt
        return torch.cat(outs, dim=1)


# -- inference wrapper -----------------------------------------------------------
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

    # encode / decode --------------------------------------------------------
    @torch.no_grad()
    def encode(self, smiles_list):
        """List of SMILES -> latent means, np array (n, latent). Invalid rows dropped.

        Returns (Z, kept_smiles) so callers know which inputs survived encoding.
        """
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
        """Latents Z (n, latent) -> list of canonical SMILES (None where invalid)."""
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

    # sampling ---------------------------------------------------------------
    def sample_latents(self, n, anchors=None, scale=1.0, jitter=0.5, rng=None):
        """Draw n candidate latents.

        With no anchors: samples from the prior N(0, scale^2 I). With anchors (an
        (m, latent) array of encoded current molecules): mixes prior samples with
        jittered copies of anchors so search stays near known-good regions while still
        exploring. Reproducible via rng.
        """
        rng = rng or np.random.default_rng()
        d = self.latent_dim
        if anchors is None or len(anchors) == 0:
            return rng.normal(0.0, scale, size=(n, d))
        anchors = np.asarray(anchors, dtype=float)
        # Explore mostly around the known-good anchors (local trust region), with a
        # smaller prior fraction for global exploration.
        near_n = int(round(0.75 * n))
        prior = rng.normal(0.0, scale, size=(n - near_n, d))
        idx = rng.integers(0, len(anchors), size=near_n)
        near = anchors[idx] + rng.normal(0.0, jitter, size=(near_n, d))
        return np.vstack([near, prior])

    # persistence ------------------------------------------------------------
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
