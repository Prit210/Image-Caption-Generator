
import torch, os, math
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model   import Model
from dataset import CocoDataset

device   = "cuda"
IMG      = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017"
ANN      = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/captions_train2017.json"
VAL_IMG  = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017"
VAL_ANN  = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/captions_val2017.json"
EPOCHS   = 15
BATCH    = 256
LR       = 2e-4
CKPT     = "runs/best.pt"
RESUME   = False  # smoke test: always start fresh
WARMUP   = 2    # epochs of linear LR warmup

vocab = torch.load("vocab.pt", weights_only=False)
ds    = CocoDataset(IMG, ANN, vocab, limit=150000, train=True)
val_ds = CocoDataset(VAL_IMG, VAL_ANN, vocab, limit=15000, train=False)
dl    = DataLoader(ds, BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_dl = DataLoader(val_ds, BATCH, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
print(f"Train samples: {len(ds):,} | Val samples: {len(val_ds):,}")

n_gpu = torch.cuda.device_count()
model = Model(len(vocab)).to(device)
if n_gpu > 1:
    print(f"Using {n_gpu} GPUs via DataParallel")
    model = torch.nn.DataParallel(model)
m = model.module if hasattr(model, "module") else model

start_epoch = 0
best_val    = float("inf")
if RESUME:
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    m.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    best_val    = ckpt.get("val_loss", float("inf"))
    print(f"Resumed from epoch {start_epoch}, best_val={best_val:.4f}")

def set_requires_grad(layers, value):
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = value

def make_opt(model, lr):
    # Collect proj/norm/pos/cap/proj2 first, then exclude them from enc_params
    # to avoid duplicate params across param groups (AdamW raises ValueError)
    proj_params = list(m.enc.proj.parameters()) + list(m.enc.proj2.parameters())                 + list(m.enc.norm.parameters()) + list(m.enc.pos.parameters())                 + list(m.enc.cap.parameters())
    proj_ids    = {id(p) for p in proj_params}
    enc_params  = [p for p in m.enc.parameters()
                   if p.requires_grad and id(p) not in proj_ids]
    dec_params  = list(m.dec.parameters())
    return torch.optim.AdamW([
        {"params": enc_params,  "lr": lr * 0.1},
        {"params": proj_params, "lr": lr},
        {"params": dec_params,  "lr": lr},
    ], weight_decay=1e-4)

opt     = make_opt(model, LR)
total_steps = EPOCHS * len(dl)

def lr_lambda(step):
    warmup_steps = WARMUP * len(dl)
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
if RESUME:
    # Fast-forward scheduler using last_epoch (avoids slow loop)
    done_steps = start_epoch * len(dl)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=done_steps - 1)

loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)
scaler  = torch.amp.GradScaler("cuda")

UNFREEZE = {5: [m.enc.layer3, m.enc.layer4], 10: [m.enc.layer1, m.enc.layer2]}

global_step = start_epoch * len(dl)

for e in range(start_epoch, EPOCHS):
    # Progressive unfreeze — recreate opt AND scheduler so new opt gets LR updates
    if e in UNFREEZE:
        set_requires_grad(UNFREEZE[e], True)
        opt = make_opt(model, LR)
        for g in opt.param_groups:
            g.setdefault("initial_lr", g["lr"])  # required by LambdaLR when last_epoch>=0
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda, last_epoch=global_step - 1)
        print(f"Epoch {e}: unfreezing {[type(l).__name__ for l in UNFREEZE[e]]}")

    # ── Train ─────────────────────────────────────────────────────
    model.train()
    total_loss = 0.0
    for img, cap, _ in tqdm(dl, desc=f"Train {e}"):
        img, cap = img.to(device, non_blocking=True), cap.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            out  = model(img, cap[:, :-1])
            loss = loss_fn(out.reshape(-1, out.size(-1)), cap[:, 1:].reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()
        global_step += 1
        total_loss += loss.item()

    avg_tr = total_loss / len(dl)

    # ── Val ───────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img, cap, _ in val_dl:
            img, cap = img.to(device), cap.to(device)
            with torch.amp.autocast("cuda"):
                out  = model(img, cap[:, :-1])
                loss = loss_fn(out.reshape(-1, out.size(-1)), cap[:, 1:].reshape(-1))
            val_loss += loss.item()
    avg_val = val_loss / len(val_dl)

    cur_lr = opt.param_groups[-1]["lr"]
    print(f"Epoch {e} | train={avg_tr:.4f} | val={avg_val:.4f} | lr={cur_lr:.2e}")

    if avg_val < best_val:
        best_val = avg_val
        torch.save({"model": m.state_dict(), "epoch": e, "val_loss": best_val}, CKPT)
        print(f"  ✓ Saved best (val={best_val:.4f})")
    else:
        # Always save latest too so we can resume
        torch.save({"model": m.state_dict(), "epoch": e, "val_loss": avg_val}, "runs/latest.pt")
