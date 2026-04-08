# Multimodal Fake News Detection — Fakeddit

> **Classify Reddit posts (headline + image) into 6 fake-news categories using BERT+ResNet-50, BERT+ViT, and CLIP.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset — Fakeddit](#3-dataset--fakeddit)
4. [Google Drive Folder Setup](#4-google-drive-folder-setup-required-before-anything-else)
5. [Step-by-Step: Running on Google Colab (Recommended)](#5-step-by-step-running-on-google-colab-recommended)
6. [Training Notebooks](#6-training-notebooks)
7. [Web App Demo (Streamlit)](#7-web-app-demo-streamlit)
8. [Pre-Trained Weights](#8-pre-trained-weights)
9. [Installation Challenge — Quick Checklist](#9-installation-challenge--quick-checklist)
10. [Expected Performance](#10-expected-performance)
11. [Configuration Reference](#11-configuration-reference)
12. [License](#12-license)
13. [Authors](#13-authors)

---

## 1. Project Overview

We train three multimodal classifiers on the **Fakeddit** dataset to detect fake news using both the article **headline (text)** and the accompanying **image**:

| Model | Text Encoder | Image Encoder | Checkpoint |
|---|---|---|---|
| BERT + ResNet-50 | `bert-base-uncased` | ResNet-50 (ImageNet) | `m1.pth` |
| BERT + ViT | `bert-base-uncased` | `google/vit-base-patch16-224-in21k` | `bert_vit_v2_2_best_score.pth` |
| CLIP Classifier | `openai/clip-vit-base-patch32` | CLIP Vision Tower | `clip_multimodal_bestv1.pth` |

A **Streamlit demo app** loads all three models, classifies any (headline, image) pair into one of six labels, and shows a majority-vote result.

---

## 2. Repository Structure

```
Multimodal-Fake-News-Classification-on-Fakeddit/
└── Fake-News-Multimodal-Classification/
    ├── dataset_downloader.ipynb          <- Step 1: download & prepare the dataset
    ├── requirements.txt                  <- Python dependencies
    ├── Training Notebooks/
    │   ├── bertandrestnet.ipynb          <- Train BERT + ResNet-50
    │   ├── bert_ViT_v2.ipynb             <- Train BERT + ViT
    │   ├── CLIPv1.ipynb                  <- CLIP experiments / evaluation
    │   └── CLIPv2_1.ipynb                <- Train final CLIP classifier
    └── Fakeddit-WebApp/
        ├── RESVITCLIPv1/                 <- Ensemble app (all 3 models)
        │   ├── app.py
        │   └── Model.py
        └── CLIPv2/                       <- CLIP-only app
            ├── app.py
            └── Model.py
```

---

## 3. Dataset — Fakeddit

[Fakeddit](https://github.com/entitize/Fakeddit) is a large-scale multimodal fake news dataset built from Reddit. We use the **6-way label** split:

| Label ID | Label Name | Description |
|---|---|---|
| 0 | TRUE | Factually accurate, verified content |
| 1 | SATIRE | Humorous/satirical content, not meant literally |
| 2 | FALSE CONNECTION | Headline/image does not match the actual content |
| 3 | IMPOSTER CONTENT | Impersonates a genuine news source |
| 4 | MANIPULATED CONTENT | Edited/altered content to mislead |
| 5 | MISLEADING CONTENT | Selective framing or missing context |

### Fakeddit TSV Files

You need **3 TSV files** from the official Fakeddit release:

- `multimodal_train.tsv`
- `multimodal_test_public.tsv`
- `multimodal_validate.tsv`

Download them from the [Fakeddit-Data Kaggle](https://www.kaggle.com/datasets/vanshikavmittal/fakeddit-dataset/data) (provided by the dataset owners).

---

## 4. Google Drive Folder Setup (required before anything else)

> ⚠️ **Critical:** Google Colab runtimes reset when disconnected. All downloaded images and generated files **must live in Google Drive** so they persist across sessions. Do not store them only in `/content` — that disk is wiped on every disconnect.

### 4.1 Create the Folder Tree in Your Google Drive

Go to [drive.google.com](https://drive.google.com) and create the following folder structure **exactly as shown** — spelling and capitalisation matter, as the notebooks use these exact paths:

```
My Drive/
└── fake-news-detector/
    └── multimodal_only_samples/
        └── working/
            └── images/        <- images will be downloaded here automatically
```

### 4.2 Upload the 3 TSV Files

Upload the three TSV files directly inside `My Drive/fake-news-detector/multimodal_only_samples/` (**not** inside `working/`):

```
My Drive/fake-news-detector/multimodal_only_samples/
    ├── multimodal_train.tsv
    ├── multimodal_test_public.tsv
    ├── multimodal_validate.tsv
    └── working/
        └── images/            <- empty for now
```

Once the dataset downloader runs (Step 5.3), it will populate `working/images/` with ~18,000 images and save `working/clean_df.csv`.

---

## 5. Step-by-Step: Running on Google Colab (Recommended)

### 5.1 Clone the Repository Inside Colab

Open a **new Colab notebook** and run:

```python
!git clone https://github.com/ak-saxena/Multimodal-Fake-News-Classification-on-Fakeddit.git
```

This places the repo at `/content/Multimodal-Fake-News-Classification-on-Fakeddit/`. Verify with:

```python
import os
print(os.listdir('/content/Multimodal-Fake-News-Classification-on-Fakeddit/Fake-News-Multimodal-Classification/'))
# Expected: ['dataset_downloader.ipynb', 'requirements.txt', 'Training Notebooks', 'Fakeddit-WebApp', ...]
```

### 5.2 Install Dependencies

```python
!pip install -q \
  torch torchvision torchaudio \
  transformers \
  accelerate \
  scikit-learn \
  pandas \
  numpy \
  pillow \
  matplotlib \
  tqdm \
  streamlit
```

> PyTorch comes pre-installed on Colab with CUDA support. The command above upgrades/installs the remaining packages. You can also install from the repo's `requirements.txt`:
> ```python
> !pip install -q -r /content/Multimodal-Fake-News-Classification-on-Fakeddit/Fake-News-Multimodal-Classification/requirements.txt
> ```

### 5.3 Mount Google Drive and Set Up the Dataset

#### 5.3.1 Mount Your Drive

Run this at the start of **every Colab session** (the notebooks include this cell already):

```python
from google.colab import drive
drive.mount('/content/drive')
```

After authenticating, your Drive is accessible at `/content/drive/MyDrive/`.

#### 5.3.2 Verify the Folder Structure

```python
import os

base = "/content/drive/MyDrive/fake-news-detector/multimodal_only_samples"
print(os.listdir(base))
# Expected: ['multimodal_train.tsv', 'multimodal_test_public.tsv', 'multimodal_validate.tsv', 'working']

print(os.listdir(os.path.join(base, "working")))
# Expected: ['images']  (empty at first; images appear after the downloader runs)
```

#### 5.3.3 Run `dataset_downloader.ipynb`

In Colab, open the notebook via the left-side file browser at:

```
/content/Multimodal-Fake-News-Classification-on-Fakeddit/Fake-News-Multimodal-Classification/dataset_downloader.ipynb
```

Or navigate to it via `File → Open notebook → Upload`.

**What the notebook does — in order:**

1. Mounts Google Drive.
2. Reads and merges the 3 TSV files from `multimodal_only_samples/`.
3. Filters rows to keep only posts that have images and valid 6-way labels.
4. Downloads all images from `image_url` into `working/images/` — each saved as `<id>.jpg`.
5. Logs any failed/skipped image downloads.
6. Saves the cleaned dataframe as **`working/clean_df.csv`** (~18,470 rows, 13 columns).

> ⏱ **Image downloading is the slow step** — expect 1–3 hours depending on network speed. The notebook shows a `tqdm` progress bar. Already-downloaded images are skipped on re-run, so you can safely stop and resume. Everything saves to Drive so nothing is lost on Colab disconnect.

After completion, your Drive will contain:

```
My Drive/fake-news-detector/multimodal_only_samples/working/
    ├── clean_df.csv          <- ~18,470 rows, 13 columns
    └── images/
        ├── 53y2yj.jpg
        ├── 8toxfk.jpg
        └── ...               <- ~18,000 images
```

### 5.4 Run a Training Notebook

All training notebooks are inside `Training Notebooks/`. Open any one of them in Colab.

Before running:

1. **Enable GPU:** `Runtime → Change runtime type → T4 GPU` (or A100 via Colab Pro).
2. **Check the data paths** at the top of each notebook:

```python
CLEANDFPATH = "/content/drive/MyDrive/fake-news-detector/multimodal_only_samples/working/clean_df.csv"
IMAGEDIR    = "/content/drive/MyDrive/fake-news-detector/multimodal_only_samples/working/images"
```

If your Google Drive folder is named differently, update these two constants. Then run all cells top-to-bottom.

Each notebook saves its best checkpoint directly to Drive so it survives session resets.

---

## 6. Training Notebooks

### 6.1 BERT + ResNet-50 (`bertandrestnet.ipynb`)

**Architecture:**
- **Text branch:** `bert-base-uncased` → `[CLS]` pooled output → dropout → `Linear(768 → 6)`
- **Image branch:** `ResNet-50` (ImageNet pretrained) → dropout → `Linear(1000 → 6)`
- **Fusion:** element-wise `torch.max` of the two 6-dim logit vectors → CrossEntropy loss

**Image preprocessing:** resize `256×256`, convert to tensor, normalize with ImageNet mean/std `([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`.

**Output checkpoint:** `m1.pth` → used by the ensemble app.

---

### 6.2 BERT + ViT (`bert_ViT_v2.ipynb`)

**Architecture:**
- **Text branch:** `bert-base-uncased` → BERT CLS output
- **Image branch:** `google/vit-base-patch16-224-in21k` → ViT CLS output
- **Fusion:** concatenate text + image embeddings → dropout → `Linear(→ 6)`, CrossEntropy with label smoothing

**Key hyperparameters:**

```python
MAX_LEN        = 80        # BERT token max length
BATCH_SIZE     = 16
NUM_EPOCHS     = 12
HEAD_LR        = 1e-4      # classification head learning rate
BACKBONE_LR    = 5e-6      # BERT + ViT backbone learning rate
DROPOUT        = 0.4
WEIGHT_DECAY   = 1e-4
EARLY_STOPPING_PATIENCE = 4
```

**Extras:** class weights for imbalance, mixed precision (`autocast` + `GradScaler`), cosine LR scheduler, early stopping, saves train/val/test splits as CSVs.

**Output checkpoints:**
- `bert_vit_v2_2_best_score.pth` ← best validation checkpoint, used by demo app
- `bert_vit_v2_2_final.pth` ← last epoch checkpoint

---

### 6.3 CLIP Classifier (`CLIPv2_1.ipynb`)

> `CLIPv1.ipynb` contains early experiments and evaluation tables. `CLIPv2_1.ipynb` is the **main training notebook**.

**Architecture:**
- **Backbone:** `openai/clip-vit-base-patch32` (optionally frozen)
- **Inputs:** CLIP text tokens (`input_ids`, `attention_mask`) + CLIP `pixel_values`
- **Head:** dropout (`0.2`) → `Linear(→ 6)`, CrossEntropy loss

**Output checkpoint:** `clip_multimodal_bestv1.pth` ← used by both apps.

---

## 7. Web App Demo (Streamlit)

Both apps accept a headline + image and return a 6-class prediction with confidence scores.

### 7.1 Ensemble App — All 3 Models (`Fakeddit-WebApp/RESVITCLIPv1/app.py`)

Loads BERT+ResNet, BERT+ViT, and CLIP. Shows individual predictions and a **majority vote**.

**Required files in the same folder as `app.py`:**

```
Model.py
m1.pth
bert_vit_v2_2_best_score.pth
clip_multimodal_bestv1.pth
```

**Run locally:**

```bash
cd Fake-News-Multimodal-Classification/Fakeddit-WebApp/RESVITCLIPv1
streamlit run app.py
# Open: http://localhost:8501
```

### 7.2 CLIP-Only App (`Fakeddit-WebApp/CLIPv2/app.py`)

Minimal single-model demo.

**Required files:**

```
Model.py
clip_multimodal_best.pth      # update MODEL_PATH in app.py if your filename differs
```

**Run locally:**

```bash
cd Fake-News-Multimodal-Classification/Fakeddit-WebApp/CLIPv2
streamlit run app.py
```

### 7.3 Running Streamlit on Colab

Streamlit cannot open a browser tab natively from Colab. Use `pyngrok` to get a public URL:

```python
!pip install -q streamlit pyngrok
from pyngrok import ngrok
import subprocess

proc = subprocess.Popen([
    "streamlit", "run",
    "/content/Multimodal-Fake-News-Classification-on-Fakeddit/Fake-News-Multimodal-Classification/Fakeddit-WebApp/RESVITCLIPv1/app.py",
    "--server.port", "8501"
])

public_url = ngrok.connect(8501)
print("Open this URL in your browser:", public_url)
```

---

## 8. Pre-Trained Weights

To run the demo **without training from scratch**, download our pre-trained checkpoints and place them in the app folder:

| Checkpoint | Model | Download |
|---|---|---|
| `m1.pth` | BERT + ResNet-50 | *(https://drive.google.com/drive/folders/1cHm_FllwuDGQLDTGxbLGGmJ9ty1QeKAi?usp=sharing)* |
| `bert_vit_v2_2_best_score.pth` | BERT + ViT | *(https://drive.google.com/drive/folders/1cHm_FllwuDGQLDTGxbLGGmJ9ty1QeKAi?usp=sharing)* |
| `clip_multimodal_bestv1.pth` | CLIP Classifier v1| *(https://drive.google.com/drive/folders/1cHm_FllwuDGQLDTGxbLGGmJ9ty1QeKAi?usp=sharing)* |

> **During the lab session:** we will share the checkpoint files directly. Place them next to `app.py` and the app will load them automatically.

---

## 9. Installation Challenge — Quick Checklist

For another group completing the **30-minute lab installation**:

```
[1]. Create Google Drive folder tree (Section 4.1)
       My Drive/fake-news-detector/multimodal_only_samples/working/images/

[2]. Upload the 3 TSV files into multimodal_only_samples/ (Section 4.2)
       multimodal_train.tsv, multimodal_test_public.tsv, multimodal_validate.tsv

[3]. Open a new Colab notebook and clone the repo (Section 5.1):
       !git clone https://github.com/ak-saxena/Multimodal-Fake-News-Classification-on-Fakeddit.git

[4]. Install dependencies (Section 5.2)

[5]. Mount Drive and run dataset_downloader.ipynb (Section 5.3)
       -> For the lab, you only need clean_df.csv + a partial image set to
          verify training notebooks can start. Full download takes 1-3 hours
          and can run in the background.

[6]. Get pre-trained checkpoints from the authors (Section 8)
       Place them next to the corresponding app.py

[7]. Run the demo app (Section 7.1):
       streamlit run app.py
       (or use pyngrok if running on Colab — Section 7.3)

[8]. Test: enter any headline + upload an image
       -> verify predictions and majority vote appear correctly
```

If anything breaks, contact us — we can fix paths or update code on the fly during the session.

---

## 10. Best Performance

Results on the Fakeddit test set (CLIP classifier V2, from `CLIPv2_1.ipynb`):

| Metric | Value |
|---|---|
| Test Loss | 1.2882 |
| Test Accuracy | **83.22%** |
| Weighted Precision | 83.04% |
| Weighted Recall | 83.22% |
| Weighted F1 | 83.00% |
| Macro F1 | 73.86% |

Per-class breakdown (test set, n = 1847):

| Label | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| TRUE (0) | 0.8527 | 0.9119 | 0.8813 | 1022 |
| SATIRE (1) | 0.7943 | 0.7089 | 0.7492 | 158 |
| FALSE CONNECTION (2) | 0.8319 | 0.7673 | 0.7983 | 490 |
| IMPOSTER CONTENT (3) | 0.4528 | 0.4211 | 0.4364 | 57 |
| MANIPULATED CONTENT (4) | 0.8333 | 0.6667 | 0.7407 | 15 |
| MISLEADING CONTENT (5) | 0.8646 | 0.7905 | 0.8259 | 105 |

---

## 11. Configuration Reference

All hard-coded paths and hyperparameters are defined near the **top of each notebook or `app.py`**. Update them if your folder structure differs.

| File | Variable | Default Value |
|---|---|---|
| All training notebooks | `CLEANDFPATH` | `/content/drive/MyDrive/fake-news-detector/multimodal_only_samples/working/clean_df.csv` |
| All training notebooks | `IMAGEDIR` | `/content/drive/MyDrive/fake-news-detector/multimodal_only_samples/working/images` |
| `bert_ViT_v2.ipynb` | `SAVE_DIR` | `.../multimodal_only_samples/bert_vit_v2_2` |
| `RESVITCLIPv1/app.py` | checkpoint names | `m1.pth`, `bert_vit_v2_2_best_score.pth`, `clip_multimodal_bestv1.pth` |
| `CLIPv2/app.py` | `MODEL_PATH` | `clip_multimodal_best.pth` |

---

## 12. License

This project is released under the **[GNU AGPL v3.0](LICENSE)** license.

---

## 13. Authors

| Name | Email |
|---|---|
| Akshit Saxena | saxenaak@tcd.ie |
| Naysha Kumari | nkumari@tcd.ie |

