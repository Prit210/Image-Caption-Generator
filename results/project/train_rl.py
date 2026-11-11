
import torch, json, math
import torch.nn as nn
from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm
from model   import Model
from dataset import CocoDataset

device = "cuda"
IMG    = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/train2017"
ANN    = "/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/captions_train2017.json"
BATCH  = 48     # larger batch for speed (effective 96 with ACCUM=2)
ACCUM  = 2      # gradient accumulation → effective batch = 64
RL_LR  = 5e-6
EPOCHS = 5

vocab     = torch.load("vocab.pt", weights_only=False)
inv_vocab = {v: k for k, v in vocab.items()}

ds = CocoDataset(IMG, ANN, vocab, limit=50000, train=True)
dl = DataLoader(ds, BATCH, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

model = Model(len(vocab)).to(device)
n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    model = torch.nn.DataParallel(model)
    device_ids = list(range(n_gpu))
else:
    device_ids = [0]
m = model.module if hasattr(model, "module") else model
m.load_state_dict(torch.load("runs/best.pt", weights_only=False)["model"])

opt    = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), RL_LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda")

ann_data = json.load(open(ANN))
gt_caps  = {}
for a in ann_data["annotations"]:
    gt_caps.setdefault(a["image_id"], []).append(a["caption"])


def ids_to_str(ids):
    words = [inv_vocab.get(i, "") for i in ids if i not in (0, 1, 2, 3)]
    return " ".join(w for w in words if w).strip()


def ngrams(tokens, n):
    return Counter(zip(*[tokens[i:] for i in range(n)]))


def bleu4_score(hyp_toks, refs_toks):
    """Clipped BLEU-4 (corpus-level approximation per sentence)."""
    if len(hyp_toks) < 4 or not refs_toks:
        return 0.0
    bp = min(1.0, math.exp(1 - min(len(r) for r in refs_toks if r) / max(len(hyp_toks), 1)))
    score = 1.0
    for n in range(1, 5):
        h_ng  = ngrams(hyp_toks, n)
        best  = Counter()
        for r in refs_toks:
            r_ng = ngrams(r, n)
            for g in h_ng:
                best[g] = max(best[g], r_ng[g])
        clipped = sum(min(c, best[g]) for g, c in h_ng.items())
        total   = sum(h_ng.values())
        if total == 0:
            return 0.0
        score  *= (clipped / total)
    return bp * (score ** 0.25)


def build_idf(gt_caps_dict, ns=(1, 2)):
    doc_freq = {n: Counter() for n in ns}
    n_docs   = 0
    for caps in gt_caps_dict.values():
        n_docs += 1
        for cap in caps:
            toks = cap.lower().split()
            for n in ns:
                doc_freq[n].update(set(ngrams(toks, n).keys()))
    idf = {}
    for n in ns:
        for gram, df in doc_freq[n].items():
            idf[gram] = math.log((n_docs + 1.0) / (df + 1.0))
    return idf


print("Building IDF weights (~30s)...")
IDF = build_idf(gt_caps)
print(f"IDF vocab: {len(IDF):,} n-grams")


def cider_reward(hyp, refs):
    if not hyp:
        return 0.0
    h_toks = hyp.lower().split()
    score  = 0.0
    for n in (1, 2):
        h_ng   = ngrams(h_toks, n)
        if not h_ng: continue
        h_idf  = sum(IDF.get(g, 0.0) * c for g, c in h_ng.items())
        if h_idf == 0: continue
        best_f1 = 0.0
        for ref in refs:
            r_toks = ref.lower().split()
            r_ng   = ngrams(r_toks, n)
            r_idf  = sum(IDF.get(g, 0.0) * c for g, c in r_ng.items())
            if r_idf == 0: continue
            common = sum(IDF.get(g, 0.0) * min(h_ng[g], r_ng[g]) for g in h_ng if g in r_ng)
            p, r_  = common / h_idf, common / r_idf
            if p + r_ > 0:
                best_f1 = max(best_f1, 2 * p * r_ / (p + r_))
        score += best_f1
    return score / 2.0


def combined_reward(hyp, refs):
    """0.5*CIDEr + 0.5*BLEU4 — directly optimises both metrics."""
    h_toks = hyp.lower().split()
    r_toks = [r.lower().split() for r in refs]
    cider  = cider_reward(hyp, refs)
    bleu4  = bleu4_score(h_toks, r_toks)
    return 0.5 * cider + 0.5 * bleu4


def rollout(m, enc, greedy=False, max_len=24):  # match MAX_LEN=24
    B      = enc.size(0)
    seq    = torch.ones(B, 1, dtype=torch.long, device=enc.device)
    done   = torch.zeros(B, dtype=torch.bool,  device=enc.device)
    log_ps = []
    for _ in range(max_len):
        out   = m.dec(enc, seq)
        logit = out[:, -1].float()
        if greedy:
            tok = logit.argmax(-1, keepdim=True)
        else:
            probs = logit.softmax(-1)
            tok   = torch.multinomial(probs, 1)
            lp    = torch.log(probs.gather(1, tok) + 1e-9).squeeze(1)
            lp    = torch.where(done, torch.zeros_like(lp), lp)
            log_ps.append(lp)
        tok  = torch.where(done.unsqueeze(1), torch.zeros_like(tok), tok)
        seq  = torch.cat([seq, tok], dim=1)
        done = done | (tok.squeeze(1) == 2)
        if done.all(): break
    seqs = seq[:, 1:].tolist()
    if greedy:
        return seqs
    T = max_len
    stacked = torch.stack(log_ps, dim=1) if log_ps else torch.zeros(B, T, device=enc.device)
    if stacked.size(1) < T:
        pad     = torch.zeros(B, T - stacked.size(1), device=enc.device)
        stacked = torch.cat([stacked, pad], dim=1)
    return seqs, stacked


best_reward = -float("inf")
for e in range(EPOCHS):
    model.eval()
    total_r = 0.0
    opt.zero_grad(set_to_none=True)

    for step, (img, cap, img_ids) in enumerate(tqdm(dl, desc=f"RL {e}")):
        img       = img.to(device, non_blocking=True)
        refs_list = [gt_caps.get(iid.item(), [""]) for iid in img_ids]

        with torch.amp.autocast("cuda"):
            enc = m.enc(img) if n_gpu == 1 else nn.parallel.data_parallel(m.enc, img, device_ids)
            enc_d = enc.detach()

            with torch.no_grad():
                g_seqs  = rollout(m, enc_d, greedy=True)
            g_strs      = [ids_to_str(s) for s in g_seqs]
            r_base      = [combined_reward(h, r) for h, r in zip(g_strs, refs_list)]

            s_seqs, lps = rollout(m, enc, greedy=False)
            lps = lps.float()  # ensure fp32 — rollout inside autocast may return fp16
            s_strs      = [ids_to_str(s) for s in s_seqs]
            r_samp      = [combined_reward(h, r) for h, r in zip(s_strs, refs_list)]

            raw = torch.tensor([rs - rb for rs, rb in zip(r_samp, r_base)],
                               device=device, dtype=torch.float32)
            total_r += raw.mean().item()
            rewards  = (raw - raw.mean()) / (raw.std() + 1e-9)
            loss     = -(rewards.unsqueeze(1) * lps).sum(1).mean() / ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

    avg_r = total_r / len(dl)
    print(f"RL Epoch {e} | avg_reward={avg_r:.4f}")
    if avg_r > best_reward:
        best_reward = avg_r
        torch.save({"model": m.state_dict(), "epoch": e}, "runs/best_rl.pt")
        print(f"  ✓ Saved best (reward={best_reward:.4f})")
