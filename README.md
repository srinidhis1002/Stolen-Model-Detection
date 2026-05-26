# Stolen-Model-Detection
 
**Team Members:** Mahitha Senthilnathan · Srinidhi Sreenivasan<br>
**Matriculation Numbers:** 7085415 · 7086703  
**CMS Team ID:** 82(LXXX) 

---

## What This Notebook Does

Given a target ResNet-18 model trained on CIFAR-100 and 360 suspect models, the notebook ranks the suspects by their likelihood of having been stolen from the target. It extracts 22 features per suspect model (weight similarity, behavioural agreement, representational similarity via CKA) and combines three unsupervised anomaly detectors (Isolation Forest, One-Class SVM, weighted linear scorer) into an ensemble. The output is a `submission.csv` file with a suspicion score for each suspect model.

---

## Requirements

Python 3.8+ and the following packages:

```bash
pip install torch torchvision safetensors huggingface_hub scikit-learn scipy tqdm pandas numpy
```

A **GPU is strongly recommended** : feature extraction across 360 models is slow on CPU. The notebook auto-detects CUDA and falls back to CPU if unavailable.

---

## How to Run

### Option 1 — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/) and upload `Assignment_2.ipynb` via **File → Upload notebook**.
2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.
3. Run the first cell : it will install all dependencies automatically.
4. Run all remaining cells in order (**Runtime → Run all**).
5. The output file `submission.csv` will be saved in the Colab working directory (`/content/`). Download it via the Files panel on the left.

> **Optional:** To persist the downloaded model cache across Colab sessions, mount Google Drive before running:
> ```python
> from google.colab import drive
> drive.mount("/content/drive")
> ```
> The notebook already contains this snippet , just uncomment it in the config cell.

### Option 2 — Local Jupyter

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install torch torchvision safetensors huggingface_hub scikit-learn scipy tqdm pandas numpy
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook Assignment_2.ipynb
   ```
4. Run all cells in order (**Cell → Run All**).
5. `submission.csv` will be saved in the same directory as the notebook.

---

## What Gets Downloaded Automatically

The notebook downloads the following from HuggingFace (`SprintML/tml26_task2`) on first run:

- Target model weights (`target_model/weights.safetensors`)
- Training index file (`target_model/train_main_idx.json`)
- All 360 suspect model weights

CIFAR-100 is also downloaded automatically via `torchvision.datasets.CIFAR100`.

All files are cached locally in `./hf_cache/` and `./cifar100_data/` so subsequent runs are faster.

---

## Output

| File | Description |
|---|---|
| `submission.csv` | Two columns: `id` (suspect model index) and `score` (suspicion score, higher = more likely stolen) |

---

## Approximate Runtime

| Hardware | Estimated Time |
|---|---|
| Colab T4 GPU | ~25–40 minutes |
| Local GPU (RTX 3080 or similar) | ~20–30 minutes |
| CPU only | ~3–5 hours |
