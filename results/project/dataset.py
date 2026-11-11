
import json, torch, random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = T.Compose([
    T.Resize((288, 288)),
    T.RandomCrop(256),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=9),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

MAX_LEN = 24   # <start> + 22 content tokens + <end>  (COCO avg ~10 tokens; 24 sufficient)

class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, vocab, limit=None, train=True):
        self.img_dir = img_dir
        self.vocab   = vocab
        self.t       = TRAIN_TRANSFORM if train else VAL_TRANSFORM
        data = json.load(open(ann_file))

        if train:
            # Use every annotation as a separate sample — 414k vs 50k
            self.samples = [(a["image_id"], a["caption"]) for a in data["annotations"]]
            if limit:
                random.shuffle(self.samples)
                self.samples = self.samples[:limit]
        else:
            # Val: one sample per image (we pick one cap; eval uses gt file)
            caps = {}
            for a in data["annotations"]:
                caps.setdefault(a["image_id"], []).append(a["caption"])
            ids = list(caps.keys())
            if limit:
                ids = ids[:limit]
            self.samples = [(iid, caps[iid][0]) for iid in ids]
            self.caps = caps   # for reference access

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img_id, cap = self.samples[i]
        path = f"{self.img_dir}/{str(img_id).zfill(12)}.jpg"
        img  = Image.open(path).convert("RGB")
        img  = self.t(img)
        tokens = cap.lower().split()
        ids = [1] + [self.vocab.get(w, 3) for w in tokens[:MAX_LEN - 2]] + [2]
        ids += [0] * (MAX_LEN - len(ids))
        return img, torch.tensor(ids), img_id
