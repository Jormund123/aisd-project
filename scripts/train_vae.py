"""
Train the small SELFIES-VAE on AqSolDB and save a checkpoint for the generative
latent-space BO in Approaches 2 and 3.

This is the one-time "pretrained encoder-decoder" the mentor's loop needs. The VAE only
learns the chemistry of the molecule space (how to encode/decode valid molecules); it
never sees solubility labels. LogS enters later, in the BO loop, via the LLM surrogate.

Run from the repo root (CPU is fine, a few minutes):
    ./venv/bin/python scripts/train_vae.py --epochs 12
Outputs: models/selfies_vae.pt
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from esol import validate_smiles          # noqa: E402
from vae import (                          # noqa: E402
    VAEModel, SelfiesVAE, PAD, BOS, EOS,
    smiles_to_selfies, build_vocab, selfies_to_ids,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "aqsoldb.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "selfies_vae.pt")


def load_selfies(limit, max_len, seed):
    """Return SELFIES strings for valid AqSolDB molecules within the length cap."""
    df = pd.read_csv(DATA)
    smiles = df["SMILES"].astype(str).tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(smiles)
    kept = []
    import selfies as sf
    for smi in smiles:
        mol, canonical = validate_smiles(smi)
        if mol is None or "." in canonical:
            continue
        s = smiles_to_selfies(canonical)
        if s is None:
            continue
        if len(list(sf.split_selfies(s))) > max_len - 2:
            continue
        kept.append(s)
        if limit and len(kept) >= limit:
            break
    return kept


def token_accuracy(logits, target, pad_idx):
    """Fraction of non-pad target tokens predicted correctly (teacher forced)."""
    pred = logits.argmax(-1)
    mask = target != pad_idx
    return (pred[mask] == target[mask]).float().mean().item()


def main():
    p = argparse.ArgumentParser(description="Train the SELFIES-VAE")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--latent", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--max-len", type=int, default=72)
    p.add_argument("--limit", type=int, default=0, help="0 = use all valid molecules")
    p.add_argument("--beta", type=float, default=0.5, help="max KL weight (annealed)")
    p.add_argument("--free-bits", type=float, default=0.1,
                   help="min nats each latent dim must carry (prevents collapse)")
    p.add_argument("--word-dropout", type=float, default=0.4,
                   help="prob of blanking a teacher-forced token (forces z reliance)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading + tokenizing AqSolDB ...")
    selfies_list = load_selfies(args.limit, args.max_len, args.seed)
    tokens, stoi = build_vocab(selfies_list)
    pad_idx = stoi[PAD]
    print(f"  molecules: {len(selfies_list)} | vocab: {len(tokens)} | max_len {args.max_len}")

    X = torch.tensor([selfies_to_ids(s, stoi, args.max_len) for s in selfies_list],
                     dtype=torch.long)

    config = {"emb": args.emb, "hidden": args.hidden, "latent": args.latent,
              "max_len": args.max_len}
    model = VAEModel(len(tokens), emb=args.emb, hidden=args.hidden,
                     latent=args.latent, pad_idx=pad_idx)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    n = len(X)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        beta = args.beta * min(1.0, epoch / max(1, args.epochs // 2))  # KL warmup
        tot_ce = tot_kl = tot_acc = 0.0
        nb = 0
        for start in range(0, n, args.batch):
            xb = X[perm[start:start + args.batch]]
            # Word dropout: randomly blank teacher-forced input tokens (-> PAD embedding
            # = zero vector), so the decoder must lean on z rather than the copied token.
            dec_in = xb[:, :-1].clone()
            if args.word_dropout > 0:
                drop = torch.rand_like(dec_in, dtype=torch.float) < args.word_dropout
                drop[:, 0] = False               # keep the leading BOS
                dec_in[drop] = pad_idx
            logits, mu, logvar = model(xb, dec_input=dec_in)
            target = xb[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, len(tokens)),
                                 target.reshape(-1), ignore_index=pad_idx)
            # Free-bits KL: penalize only the info each latent dim carries ABOVE the
            # floor, so no dimension is driven to zero (guards against posterior
            # collapse, which would make decoded molecules ignore z and kill the BO).
            kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(0)  # (latent,)
            kl = kl_dim.sum()
            kl_penalty = torch.clamp(kl_dim, min=args.free_bits).sum()
            loss = ce + beta * kl_penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_ce += ce.item(); tot_kl += kl.item()
            tot_acc += token_accuracy(logits, target, pad_idx); nb += 1
        print(f"epoch {epoch:2d}/{args.epochs} | CE {tot_ce/nb:.3f} "
              f"| KL {tot_kl/nb:.3f} | beta {beta:.3f} | tok-acc {tot_acc/nb:.3f}")

    vae = SelfiesVAE(model, tokens, config, device="cpu")
    vae.save(args.out)
    print(f"\nSaved: {args.out}")

    # Sanity 1: decode prior samples and report how many are valid molecules.
    Z = vae.sample_latents(50, rng=np.random.default_rng(args.seed))
    decoded = vae.decode(Z)
    valid = [s for s in decoded if s]
    print(f"\nPrior-sample decode validity: {len(valid)}/50")
    for s in valid[:5]:
        print("   sample:", s)

    # Sanity 2: reconstruction. Encode real molecules to mu, decode, compare. Confirms
    # the latent actually controls the output (no posterior collapse) and that nearby
    # molecules stay distinct -- both required for BO to work in z-space.
    from esol import validate_smiles as _vs
    probes = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(O)=O", "OCC(O)CO", "c1ccccc1"]
    canon = [_vs(s)[1] for s in probes]
    Zc, kept = vae.encode(canon)
    recon = vae.decode(Zc)
    exact = sum(1 for a, b in zip(kept, recon) if b and _vs(b)[1] == a)
    print(f"\nReconstruction exact-match: {exact}/{len(kept)}")
    for a, b in zip(kept, recon):
        print(f"   {a:32s} -> {b}")


if __name__ == "__main__":
    main()
