# 🧠 Deep Learning Image Caption Generator

An end-to-end Image Captioning system built with PyTorch and deployed with Flask. The model combines a deep **ResNet-101** visual feature extractor, multi-scale feature projection, learnable sinusoidal 2D positional encodings, and a **Transformer Decoder**, fine-tuned with **Self-Critical Sequence Training (SCST)** reinforcement learning on the MS-COCO 2017 dataset.

---

## 🚀 Key Features

- **Hybrid CNN-Transformer Architecture**:
  - **Visual Encoder**: ResNet-101 extracting multi-level representations (layers 2, 3, and 4) combined with single-query cross-attention pooling and 2D sinusoidal positional embeddings.
  - **Language Decoder**: 4-layer Transformer Decoder with Pre-LN, multi-head attention, and weight-tied embeddings.
- **Two-Stage Training Pipeline**:
  1. **Supervised Pretraining**: Cross-entropy training with linear learning rate warmup and progressive backbone unfreezing.
  2. **Reinforcement Learning (SCST)**: Self-Critical Sequence Training directly optimizing evaluation metrics (CIDEr / BLEU) using baseline subtraction to eliminate reward bias.
- **Inference with Beam Search**:
  - Configurable beam search with Wu et al. length penalty, tri-gram blocking (repetition avoidance), and diversity scoring.
- **Interactive Web Interface**:
  - Flask-powered web application for uploading images and generating instant captions.

---

## 📁 Repository Structure

```text
├── web_deploy/                  # Flask web deployment
│   ├── app.py                   # Flask server entrypoint
│   ├── inference.py             # Inference pipeline & model loader
│   ├── model.py                 # ResNet-101 + Transformer model architecture
│   ├── beam.py                  # Beam search with n-gram blocking
│   ├── vocab.pt                 # Vocabulary mapping dictionary
│   ├── templates/               # HTML UI templates (index.html)
│   ├── static/                  # Static assets & upload directory
│   └── runs/                    # Directory for model checkpoint (best_rl.pt)
│
├── results/                     # Training and evaluation scripts
│   └── project/
│       ├── model.py             # Model definitions
│       ├── dataset.py           # MS-COCO PyTorch Dataset loader & tokenizer
│       ├── train.py             # Phase 1: Cross-Entropy supervised training
│       ├── train_rl.py          # Phase 2: SCST Reinforcement Learning training
│       ├── evaluate.py          # BLEU, METEOR, and CIDEr evaluation script
│       ├── beam.py              # Beam search implementation
│       ├── vocab.pt             # Vocabulary mapping dictionary
│       └── results.json         # Evaluation benchmarks and sample predictions
│
├── notebooks/                   # Jupyter notebooks
│   ├── coco_image_captioning.ipynb
│   └── coco_image_captioning_with_outputs.ipynb
│
├── sample_images/               # Example images to test the model
├── requirements.txt             # Python dependencies
└── .gitignore                   # Git ignore rules (filters large weights & cache)
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 📦 Model Weights Setup

Because PyTorch checkpoint files (`*.pt`) exceed GitHub's 100 MB file size limit (~390 MB each), they are tracked via `.gitignore`.

1. Download the trained checkpoint (`best_rl.pt`).
2. Place the file inside the `web_deploy/runs/` directory:
   ```text
   web_deploy/
   └── runs/
       └── best_rl.pt
   ```

*(If you have trained weights hosted on Hugging Face Hub, Google Drive, or GitHub Releases, paste your download link here).*

---

## 🖥️ Running the Web Application

To launch the local web server:

```bash
# Navigate to web_deploy directory
cd web_deploy
python app.py
```

Or from the project root:
```bash
python web_deploy/app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

Upload any image (`.jpg`, `.png`, `.jpeg`) to generate a caption in real-time.

---

## 🏋️‍♂️ Training from Scratch

The model is trained on the [MS-COCO 2017 Dataset](https://cocodataset.org/).

### Phase 1: Cross-Entropy Training
```bash
python results/project/train.py
```
- Trains for 15 epochs with warmup.
- Saves checkpoints to `results/project/runs/best.pt`.

### Phase 2: Reinforcement Learning (SCST)
```bash
python results/project/train_rl.py
```
- Loads the best cross-entropy checkpoint.
- Fine-tunes using policy gradient with greedy baseline.
- Saves final weights to `results/project/runs/best_rl.pt`.

### Evaluation
```bash
python results/project/evaluate.py
```

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
