import os
import torch
from PIL import Image
import torchvision.transforms as T
from model import Model
from beam import beam_search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vocab_path = os.path.join(BASE_DIR, "vocab.pt")
ckpt_path = os.path.join(BASE_DIR, "runs", "best_rl.pt")

if not os.path.exists(vocab_path):
    raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")

vocab = torch.load(vocab_path, map_location="cpu")
inv_vocab = {v: k for k, v in vocab.items()}

model = Model(len(vocab)).to(DEVICE)
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
else:
    print(f"[WARNING] Checkpoint not found at {ckpt_path}. Please place 'best_rl.pt' inside '{os.path.join(BASE_DIR, 'runs')}'.")

SKIP = {0, 1, 2, 3}

transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        enc = model.enc(img)
        ids = beam_search(model, enc)

    words = [inv_vocab.get(i, "") for i in ids[1:] if i not in SKIP]
    return " ".join(words)
