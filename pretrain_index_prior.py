import os, json, argparse, math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scene.index_prior import IndexPrior


# -----------------------------
# Dataset
# -----------------------------
class ShardDataset(Dataset):
    def __init__(self, shard_dir: str):
        self.files = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir) if f.endswith(".npz")])
        if not self.files:
            raise FileNotFoundError(f"No .npz shards in {shard_dir}")
        self.manifest = None
        man = os.path.join(shard_dir, "iprior_manifest.json")
        if os.path.exists(man):
            with open(man, "r") as f:
                self.manifest = json.load(f)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx: int):
        z = np.load(self.files[idx])
        return {k: torch.from_numpy(z[k]) for k in z.files}


def collate_shard(batch):
    assert len(batch) == 1
    return batch[0]


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _acc_chunk(logits: torch.Tensor, tg_shift: torch.Tensor, msk_shift: torch.Tensor):
    # logits: (b,L-1,V) ; tg_shift/msk_shift: (b,L-1)
    pred = logits.argmax(dim=-1)
    m = (msk_shift > 0)
    correct = (pred[m] == tg_shift[m]).sum().item()
    total = int(m.sum().item())
    return correct, total


# -----------------------------
# Train / Eval (causal next-token with micro-batch)
# -----------------------------
def train_one_epoch(model, loader, optimizer, scaler, device,
                    accum_steps: int = 1, micro_bsz: int = 1024, log_interval: int = 20):
    model.train()
    total_loss, total_correct, total_tokens, steps = 0.0, 0, 0, 0
    optimizer.zero_grad(set_to_none=True)

    for it, batch in enumerate(loader):
        # Original tensors: (B,L,*) with mask 1=valid
        nb  = batch["neighbor_indices"].to(device).long()
        tg  = batch["target_indices"].to(device).long()
        pos = batch["positions"].to(device).float()
        msk = batch["key_padding_mask"].to(device).long()

        # ---- causal shift ----
        nb_in   = nb[:, :-1]              # (B,L-1)
        tg_next = tg[:, 1:]               # (B,L-1)
        pos_in  = pos[:, :-1]             # (B,L-1,3)
        m_prev  = msk[:, :-1].bool()
        m_next  = msk[:, 1:].bool()
        msk_eff = (m_prev & m_next).long()     # (B,L-1)

        B = nb_in.size(0)
        total_valid = int(msk_eff.sum().item())
        if total_valid == 0:
            continue

        grad_steps = 0
        for s in range(0, B, micro_bsz):
            e = min(s + micro_bsz, B)
            nb_s, tg_s, pos_s, msk_s = nb_in[s:e], tg_next[s:e], pos_in[s:e], msk_eff[s:e]

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                h = model.encode_context(nb_s, pos_s, msk_s)   # (b,L-1,D)
                logits = model.logits_full(h)                  # (b,L-1,V)

                b, Lm1, V = logits.shape
                # ---- fix: use reshape for non-contiguous tensors ----
                flat_logits = logits.reshape(b * Lm1, V)
                flat_tg     = tg_s.reshape(b * Lm1)
                flat_m      = (msk_s.reshape(b * Lm1) > 0)

                valid_s = int(flat_m.sum().item())
                if valid_s == 0:
                    continue

                ce_s = F.cross_entropy(flat_logits[flat_m], flat_tg[flat_m])
                # scale so sum over micro-batches equals global mean loss
                loss = ce_s * (valid_s / total_valid) / max(1, accum_steps)

            scaler.scale(loss).backward()
            grad_steps += 1

            c, t = _acc_chunk(logits.detach(), tg_s, msk_s)
            total_correct += c
            total_tokens  += t

            if grad_steps % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += (ce_s.item() * (valid_s / total_valid))

        steps += 1
        if (it + 1) % log_interval == 0:
            acc = (total_correct / max(1, total_tokens)) if total_tokens else 0.0
            print(f"[train] iter={it+1} loss={total_loss/steps:.4f} acc={acc:.4f}")

    acc = (total_correct / max(1, total_tokens)) if total_tokens else 0.0
    return {"loss": total_loss / max(1, steps), "acc": acc}


@torch.no_grad()
def evaluate(model, loader, device, micro_bsz: int = 1024, max_batches: int = 8):
    model.eval()
    total_loss, total_correct, total_tokens, steps = 0.0, 0, 0, 0

    for it, batch in enumerate(loader):
        if it >= max_batches: break

        nb  = batch["neighbor_indices"].to(device).long()
        tg  = batch["target_indices"].to(device).long()
        pos = batch["positions"].to(device).float()
        msk = batch["key_padding_mask"].to(device).long()

        nb_in   = nb[:, :-1]
        tg_next = tg[:, 1:]
        pos_in  = pos[:, :-1]
        m_prev  = msk[:, :-1].bool()
        m_next  = msk[:, 1:].bool()
        msk_eff = (m_prev & m_next).long()

        B = nb_in.size(0)
        total_valid = int(msk_eff.sum().item())
        if total_valid == 0:
            continue

        loss_sum = 0.0
        for s in range(0, B, micro_bsz):
            e = min(s + micro_bsz, B)
            nb_s, tg_s, pos_s, msk_s = nb_in[s:e], tg_next[s:e], pos_in[s:e], msk_eff[s:e]

            h = model.encode_context(nb_s, pos_s, msk_s)
            logits = model.logits_full(h)

            b, Lm1, V = logits.shape
            # ---- fix: use reshape for non-contiguous tensors ----
            flat_logits = logits.reshape(b * Lm1, V)
            flat_tg     = tg_s.reshape(b * Lm1)
            flat_m      = (msk_s.reshape(b * Lm1) > 0)

            valid_s = int(flat_m.sum().item())
            if valid_s == 0:
                continue

            ce_s = F.cross_entropy(flat_logits[flat_m], flat_tg[flat_m])
            loss_sum += ce_s.item() * (valid_s / total_valid)

            c, t = _acc_chunk(logits, tg_s, msk_s)
            total_correct += c
            total_tokens  += t

        total_loss += loss_sum
        steps += 1

    acc = (total_correct / max(1, total_tokens)) if total_tokens else 0.0
    return {"loss": total_loss / max(1, steps), "acc": acc}


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    # Required
    ap.add_argument("--shard_dir", required=True, help="Directory containing training shards")
    ap.add_argument("--out_dir",   required=True, help="Output directory for checkpoints")
    ap.add_argument("--num_codes", type=int, required=True, help="Vocabulary size of codebook")

    # Model defaults
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dim_ff", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--use_xyz", action="store_true", default=True)
    ap.add_argument("--xyz_bands", type=int, default=6)
    ap.add_argument("--xyz_scale", type=float, default=1.0)
    ap.add_argument("--causal", action="store_true", default=True)
    ap.add_argument("--tie_weights", action="store_true", default=False)

    # Training defaults
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--micro_bsz", type=int, default=1024, help="Number of sequences per micro-batch")
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_every", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seed_everything(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ShardDataset(args.shard_dir)
    if len(dataset) >= 2:
        train_set = torch.utils.data.Subset(dataset, range(len(dataset)-1))
        eval_set  = torch.utils.data.Subset(dataset, [len(dataset)-1])
    else:
        train_set, eval_set = dataset, dataset

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_shard
    )
    eval_loader  = DataLoader(
        eval_set, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_shard
    )

    model = IndexPrior(
        num_codes=args.num_codes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        use_xyz=args.use_xyz,
        xyz_bands=args.xyz_bands,
        causal=args.causal,
        tie_weights=args.tie_weights,
        xyz_scale=args.xyz_scale
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val = math.inf
    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, scaler, device,
                             accum_steps=args.accum_steps, micro_bsz=args.micro_bsz, log_interval=10)
        print(f"[epoch {epoch+1}] train loss={tr['loss']:.4f} acc={tr['acc']:.4f}")

        if (epoch+1) % args.eval_every == 0:
            ev = evaluate(model, eval_loader, device, micro_bsz=args.micro_bsz)
            print(f"[eval ] loss={ev['loss']:.4f} acc={ev['acc']:.4f}")
            if ev["loss"] < best_val:
                best_val = ev["loss"]
                torch.save(model.state_dict(), os.path.join(args.out_dir, "iprior_best.pt"))
                print(f"[save] best -> {args.out_dir}/iprior_best.pt")

        torch.save(model.state_dict(), os.path.join(args.out_dir, "iprior_last.pt"))

    print(f"[done] training complete. Best val loss={best_val:.4f}")


if __name__ == "__main__":
    main()