import torch, json, os, sys
os.chdir('/kaggle/working/project')
sys.path.insert(0, '/kaggle/working/project')
from tqdm import tqdm
from torch.utils.data import DataLoader
from model   import Model
from dataset import CocoDataset
from beam    import beam_search
import math, traceback
from collections import Counter

device = 'cuda'
IMG = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/val2017'
ANN = '/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017/annotations/captions_val2017.json'

vocab     = torch.load('vocab.pt', weights_only=False)
inv_vocab = {v: k for k, v in vocab.items()}

ds = CocoDataset(IMG, ANN, vocab, limit=500, train=False)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
print(f'Evaluating {len(ds):,} images...')

model = Model(len(vocab)).to(device)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
m = model.module if hasattr(model, 'module') else model

ckpt_path = 'runs/best_rl.pt' if os.path.exists('runs/best_rl.pt') else 'runs/best.pt'
print(f'Loading: {ckpt_path}')
m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)['model'])
model.eval()

SKIP    = {0, 1, 2, 3}
results = []

with torch.no_grad(), torch.amp.autocast('cuda'):
    for img, _, img_ids in tqdm(dl, desc='Eval'):
        img = img.to(device)
        enc = m.enc(img)
        ids = beam_search(m, enc, beam=16, max_len=30)
        words = [inv_vocab.get(i, '') for i in ids[1:] if i not in SKIP]
        cap   = ' '.join(w for w in words if w)
        results.append({'image_id': img_ids[0].item(), 'caption': cap})

with open('results.json', 'w') as f:
    json.dump(results, f)
print(f'Saved {len(results):,} captions')

try:
    from pycocotools.coco import COCO
    from pycocoevalcap.bleu.bleu   import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    coco = COCO(ANN)
    res  = coco.loadRes('results.json')
    res_ids = set(r['image_id'] for r in json.load(open('results.json')))
    coco.dataset['annotations'] = [a for a in coco.dataset['annotations'] if a['image_id'] in res_ids]
    coco.createIndex()

    gts  = {iid: [a['caption'] for a in coco.imgToAnns[iid]] for iid in res_ids}
    hyps = {iid: [res.imgToAnns[iid][0]['caption']] for iid in res_ids}

    print('\n=== Evaluation Results ===')
    bleu_scores, _ = Bleu(4).compute_score(gts, hyps)
    for i, s in enumerate(bleu_scores, 1):
        print(f'  Bleu_{i}      : {s:.4f}')
    rouge, _ = Rouge().compute_score(gts, hyps)
    print(f'  ROUGE_L      : {rouge:.4f}')
    cider, _ = Cider().compute_score(gts, hyps)
    print(f'  CIDEr        : {cider:.4f}')

except Exception as ex:
    print(f'Scorer error: {ex}')
    traceback.print_exc()

# ── Fallback: manual BLEU-1..4 ──────────────────────────────────
def _ngrams(tokens, n):
    return Counter(zip(*[tokens[i:] for i in range(n)]))

def _bleu(hyp, refs, n):
    h = hyp.lower().split()
    if len(h) < n: return 0.0
    h_ng = _ngrams(h, n)
    best = Counter()
    for r in refs:
        r_ng = _ngrams(r.lower().split(), n)
        for g in h_ng: best[g] = max(best[g], r_ng[g])
    clip = sum(min(c, best[g]) for g, c in h_ng.items())
    tot  = sum(h_ng.values())
    return clip / tot if tot else 0.0

gt = {}
for a in json.load(open(ANN))['annotations']:
    gt.setdefault(a['image_id'], []).append(a['caption'])
res_data = json.load(open('results.json'))
b1=b2=b3=b4=0.0
for r in res_data:
    refs = gt.get(r['image_id'], [''])
    hyp  = r['caption']
    h    = hyp.lower().split()
    shortest = min((len(ref.split()) for ref in refs), default=1)
    bp   = min(1.0, math.exp(1 - shortest / max(len(h), 1)))
    p1,p2,p3,p4 = [_bleu(hyp, refs, n) for n in range(1,5)]
    if all(p>0 for p in [p1,p2,p3,p4]):
        b1+=p1; b2+=p2; b3+=p3; b4+=p4
N = max(len(res_data), 1)
print('\n=== Fallback BLEU (manual) ===')
print(f'  BLEU-1: {b1/N:.4f}')
print(f'  BLEU-2: {b2/N:.4f}')
print(f'  BLEU-3: {b3/N:.4f}')
print(f'  BLEU-4: {b4/N:.4f}')
