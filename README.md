# MBG Tweet Classification — Case 1 BDC Internal 2026

> **Task**: 8-class text classification of Indonesian tweets about Program Makan Bergizi Gratis (MBG)
> **Metric**: Balanced Accuracy (macro recall — setiap kelas berkontribusi sama terlepas dari frekuensi)
> **Dataset**: 5.000 labeled tweets (train) + 1.500 tweets (test)
> **Environment**: Python 3.11 · Tesla T4 16 GB (BERT) · CPU (baseline)

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [Dataset Analysis](#dataset-analysis)
3. [Label Quality Analysis](#label-quality-analysis)
4. [Experiment Results](#experiment-results)
5. [Per-class Performance](#per-class-performance)
6. [Architecture & Methodology](#architecture--methodology)
7. [Setup & Usage](#setup--usage)
8. [Key Findings & Next Steps](#key-findings--next-steps)

---

## Project Structure

```
mbg_case1/
├── data/
│   ├── raw/                          # xlsx asli dari panitia (read-only)
│   │   ├── case_1_labeled_data.xlsx
│   │   ├── case_1_text_to_predict.xlsx
│   │   └── case_1_template_sheet.xlsx
│   ├── labeled/
│   │   └── hasil_label_baru.xlsx     # LLM re-labeled (Ollama Gemma 31B)
│   ├── cleaned/
│   │   └── labeled_clean.xlsx        # post Cleanlab manual review
│   └── outputs/
│       ├── checkpoints/              # BERT fold checkpoints (resume-safe)
│       ├── optuna/                   # Optuna study results
│       └── submissions/
│           ├── submission_baseline.xlsx       # Logistic OOF=0.6418
│           └── submission_ensemble_w0.6.xlsx  # XLM-R+IndoBERT OOF=0.7408
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis
│   ├── 02_LabelCleaning.ipynb  # Cleanlab noise detection + manual review
│   ├── 03_Baseline.ipynb       # TF-IDF + traditional ML (CPU)
│   ├── 04_AdvancedBERT.ipynb   # XLM-R + IndoBERT fine-tuning (GPU)
│   └── 05_Optuna.ipynb         # Hyperparameter tuning (Optuna)
├── src/
│   ├── config.py         # Config dataclass — semua hyperparameter
│   ├── preprocessing.py  # clean_text
│   ├── dataset.py        # TweetDataset, WeightedRandomSampler
│   ├── model.py          # MBGClassifier (CLS + mean pooling)
│   ├── losses.py         # FocalLoss + label smoothing, R-Drop KL
│   ├── optimizer.py      # LLRD optimizer, AWP
│   ├── trainer.py        # run_fold, run_cv (checkpoint resume)
│   └── utils.py          # set_seed, build_label_encoder
├── requirements.txt
├── .env.example
└── README.md
```

---

## Dataset Analysis

### Class Distribution

| Kelas | Original | LLM Re-labeled | Delta | % (LLM) |
|---|---:|---:|---:|---:|
| Kualitas Pangan | 1247 | 1229 | -18 | 24.6% |
| Politik | 792 | 780 | -12 | 15.6% |
| Anggaran | 727 | 767 | +40 | 15.3% |
| Lainnya | 638 | 570 | -68 | 11.4% |
| Tata Kelola | 511 | 569 | +58 | 11.4% |
| Sasaran Penerima | 507 | 521 | +14 | 10.4% |
| Distribusi | 433 | 368 | -65 | 7.4% |
| **Ekonomi** | **145** | **196** | **+51** | **3.9%** |
| **Total** | **5000** | **5000** | **0** | **100%** |

> **Imbalance ratio**: 1247 ÷ 145 = **8.6×** (original) → 1229 ÷ 196 = **6.3×** (LLM-labeled)
> LLM re-labeling berhasil mengurangi imbalance ratio 27%, terutama memperbaiki representasi kelas Ekonomi (+35%).

### Text Length Statistics

| Stat | Nilai |
|---|---:|
| Mean (chars) | 142.8 |
| Median (chars) | 118.0 |
| Std | 88.0 |
| Min | 7 |
| Max | 966 |
| p75 | 209 |
| p90 | 283 |
| p95 | 300 |
| p99 | 321 |

> **Implikasi untuk BERT**: `max_len=256` cukup untuk 90th percentile (283 chars ≈ ~70 tokens). Memperpanjang ke 512 hanya membantu ~5% data terpanjang dengan biaya VRAM 4×.

---

## Label Quality Analysis

Menggunakan **Cleanlab Confident Learning** untuk deteksi label noise secara algoritmik (bukan LLM, sesuai aturan kompetisi).

### Ringkasan Noise Detection

| Metric | Nilai |
|---|---:|
| Total samples | 5.000 |
| Label issues terdeteksi | **1.262 (25.2%)** |
| Sangat mencurigakan (conf < 0.3) | **1.224** |
| OOF Balanced Acc (TF-IDF SVM, pre-cleaning) | 0.5818 |

### Noise Rate per Kelas

| Kelas | Issues | Total | Noise Rate | Status |
|---|---:|---:|---:|---|
| Lainnya | 202 | 638 | **31.7%** | 🔴 HIGH |
| Politik | 248 | 792 | **31.3%** | 🔴 HIGH |
| Tata Kelola | 152 | 511 | 29.7% | 🟠 MED-HIGH |
| Sasaran Penerima | 149 | 507 | 29.4% | 🟠 MED-HIGH |
| Distribusi | 121 | 433 | 27.9% | 🟠 MED-HIGH |
| Ekonomi | 36 | 145 | 24.8% | 🟡 MEDIUM |
| Kualitas Pangan | 225 | 1247 | 18.0% | 🟡 MEDIUM |
| Anggaran | 129 | 727 | 17.7% | 🟡 MEDIUM |

> **Temuan kritis**: Kelas "Lainnya" dan "Politik" paling rentan noise. Banyak tweet yang berbau politis tapi sebenarnya membicarakan aspek spesifik (anggaran, kualitas pangan) — dan sebaliknya.

### Top Noise Confusion Pairs

| Label Saat Ini | Model Sarankan | Contoh Pola |
|---|---|---|
| Ekonomi | Tata Kelola | Tweet soal dampak ekonomi program yang diframing sebagai governance |
| Politik | Lainnya | Tweet menyebut figur politik tapi konteksnya tangensial |
| Distribusi | Kualitas Pangan | Tweet soal makanan yang sampai ke penerima (bukan distribusinya) |
| Tata Kelola | Kualitas Pangan | Kritik governance yang fokus pada insiden keracunan |

---

## Experiment Results

### Overview Progression

| Stage | Model | Strategy | OOF Balanced Acc | Gap ke 0.80 |
|---|---|---|---:|---:|
| Baseline | Logistic Regression | TF-IDF, class_weight | **0.6418** | -0.1582 |
| Baseline | SVM (LinearSVC) | TF-IDF, class_weight | 0.6365 | -0.1635 |
| Baseline | LGBM | TF-IDF + SMOTE-Tomek | 0.6042 | -0.1958 |
| Advanced | IndoBERT-p2 | LLRD + FocalLoss + R-Drop | **0.6276** | -0.1724 |
| Advanced | XLM-R-base | LLRD + FocalLoss + R-Drop | **0.7308** | -0.0692 |
| **Advanced** | **Ensemble XLM-R + IndoBERT** | **w=0.6/0.4** | **0.7408** | **-0.0592** |
| Next step | Optuna-tuned (projected) | Best HPO params | ~0.80–0.83 | — |

### Traditional ML Experiment Grid

| # | Model | Strategy | Ratio | Bal Acc | Ekonomi Recall | Runtime |
|---|---|---|---|---:|---:|---:|
| ✅ | Logistic Regression | none | 1.0 | **0.6418** | **0.6429** | 4s |
| ✅ | SVM (LinearSVC + isotonic) | none | 1.0 | 0.6365 | 0.6020 | 2s |
| | LGBM | SMOTE-Tomek | 1.0 | 0.6042 | 0.5255 | 181s |
| | LGBM | SMOTE | 0.8 | 0.5942 | 0.5306 | 138s |
| | SVM | SMOTE | 0.8 | 0.5632 | 0.3469 | 4s |
| | LGBM | none | 1.0 | 0.5576 | 0.4847 | 64s |

> **Insight**: SMOTE justru menurunkan performa SVM (0.6365 → 0.5632). Interpolasi TF-IDF sparse vectors di feature space tinggi-dimensi tidak menghasilkan synthetic samples yang valid secara semantik. `class_weight='balanced'` sudah cukup efektif untuk SVM/Logistic.

### BERT Experiment Detail

**Environment**: Tesla T4 · 15.6 GB VRAM · CUDA

#### XLM-R-base Results

| Fold | Best Epoch | Best Acc |
|---|---|---:|
| Fold 1 | 12 | ~0.71 |
| Fold 2 | ~8 | ~0.73 |
| Fold 3 | ~10 | ~0.74 |
| Fold 4 | ~9 | ~0.74 |
| Fold 5 | ~11 | ~0.73 |
| **Mean ± Std** | | **0.7308 ± 0.0194** |

#### IndoBERT-p2 Results

| Fold | Best Acc |
|---|---:|
| Fold 1 | 0.5849 |
| Fold 2–5 | ~0.62–0.66 |
| **Mean ± Std** | **0.6276 ± 0.0263** |

#### Ensemble Weight Sweep

| w_xlmr | w_indobert | Ensemble Bal Acc |
|---:|---:|---:|
| 0.0 | 1.0 | 0.6276 |
| 0.2 | 0.8 | 0.6753 |
| 0.4 | 0.6 | 0.7228 |
| **0.6** | **0.4** | **0.7408** ✅ |
| 0.8 | 0.2 | 0.7360 |
| 1.0 | 0.0 | 0.7308 |

> **Insight**: Bobot optimal w_xlmr=0.6 — XLM-R mendominasi karena lebih kuat, tapi IndoBERT tetap berkontribusi (+1.0 poin di atas XLM-R alone) karena error pattern keduanya berbeda.

---

## Per-class Performance

### Best Result: Ensemble (XLM-R 0.6 + IndoBERT 0.4) — OOF = 0.7408

| Kelas | Recall | F1 | n | Status |
|---|---:|---:|---:|---|
| Anggaran | **0.832** | 0.835 | 767 | ✅ Baik |
| Distribusi | **0.750** | 0.733 | 368 | ✅ Baik |
| Ekonomi | **0.776** | 0.766 | 196 | ✅ Baik |
| Kualitas Pangan | **0.765** | 0.805 | 1228 | ✅ Baik |
| Sasaran Penerima | 0.737 | 0.703 | 521 | ✅ Baik |
| Tata Kelola | 0.701 | 0.665 | 569 | 🟡 Perlu perhatian |
| Lainnya | 0.695 | 0.668 | 570 | 🟡 Perlu perhatian |
| Politik | 0.672 | 0.695 | 780 | 🟡 Perlu perhatian |
| **Balanced Acc** | **0.7408** | | | |

> Kelas Lainnya, Tata Kelola, dan Politik adalah tiga kelas terendah — konsisten dengan temuan Cleanlab bahwa ketiganya memiliki noise rate tertinggi (27–31%).

### Progression per Kelas (Baseline → Ensemble)

| Kelas | Logistic Baseline | XLM-R Alone | Ensemble | Δ (base→ens) |
|---|---:|---:|---:|---:|
| Tata Kelola | ~0.56 | ~0.69 | **0.701** | +~0.14 |
| Lainnya | ~0.57 | ~0.68 | **0.695** | +~0.13 |
| Ekonomi | ~0.64 | ~0.75 | **0.776** | +~0.13 |
| Sasaran Penerima | ~0.62 | ~0.72 | **0.737** | +~0.12 |
| Distribusi | ~0.64 | ~0.73 | **0.750** | +~0.11 |
| Anggaran | ~0.74 | ~0.81 | **0.832** | +~0.09 |
| Politik | ~0.58 | ~0.65 | **0.672** | +~0.09 |
| Kualitas Pangan | ~0.77 | ~0.76 | **0.765** | ~0.00 |

---

## Architecture & Methodology

### Model: MBGClassifier

```
Input tokens (max_len=256)
       ↓
XLM-R-base encoder (12 layers, hidden=768)
       ↓
CLS + Mean Pooling (concat → 1536-dim)
       ↓
Dropout(0.1)
       ↓
Linear(1536 → 8)
       ↓
Logits
```

**Kenapa CLS + Mean Pooling?**
`AutoModelForSequenceClassification` standar hanya menggunakan `[CLS]` token. Mean pooling merata-ratakan informasi dari seluruh token sequence, lebih robust untuk teks pendek (tweet) di mana representasi `[CLS]` kurang stabil.

### Training Techniques

| Teknik | Nilai | Alasan |
|---|---|---|
| **LLRD** (Layer-wise LR Decay) | factor=0.9 | Cegah catastrophic forgetting pada layer bawah encoder |
| **FocalLoss** | γ=2.0 | Down-weight easy majority-class examples |
| **Label Smoothing** | ε=0.1 | Cegah overconfidence pada kelas ambigu (Lainnya, Tata Kelola) |
| **R-Drop** | α=0.3 | KL consistency loss antar 2 forward pass → regularisasi efektif |
| **WeightedRandomSampler** | inverse-freq | Balanced batches → minority class (Ekonomi) terlihat cukup |
| **Cosine schedule** | warmup=10% | Smooth convergence, avoids loss spikes |
| **Gradient clipping** | max_norm=1.0 | Stabilitas training |
| **Checkpoint resume** | per fold | Aman dari Colab/Kaggle session timeout |

### Hyperparameter Configuration

```python
Config(
    model_name      = 'xlm-roberta-base',
    max_len         = 256,
    pooling         = 'cls_mean',
    epochs          = 12,
    patience        = 4,
    batch_size      = 16,
    grad_accum      = 2,            # effective batch = 32
    lr              = 2e-5,
    weight_decay    = 0.01,
    warmup_ratio    = 0.1,
    llrd_factor     = 0.9,
    label_smoothing = 0.1,
    focal_gamma     = 2.0,
    use_rdrop       = True,
    rdrop_alpha     = 0.3,
    scheduler       = 'cosine',
)
```

### Why XLM-R over IndoBERT?

| Aspect | IndoBERT-p2 | XLM-R-base |
|---|---|---|
| Pre-training corpus | Indonesian only | 100 languages, 2.5TB |
| Vocab size | 32.000 | 250.000 |
| OOF score (this task) | 0.6276 | **0.7308** |
| Convergence speed | Slower | Faster |
| VRAM usage | ~5 GB | ~7 GB |

> IndoBERT masih berguna dalam ensemble (+1.0 poin) karena error pattern-nya berbeda — keduanya saling complement.

---

## Setup & Usage

### Prerequisites

```bash
pip install -r requirements.txt
cp .env.example .env   # isi HF_TOKEN untuk akses model HuggingFace
```

### Data Setup

```
data/raw/case_1_labeled_data.xlsx        # ← letakkan di sini
data/raw/case_1_text_to_predict.xlsx     # ← letakkan di sini
data/raw/case_1_template_sheet.xlsx      # ← letakkan di sini
data/labeled/hasil_label_baru.xlsx       # ← hasil Ollama re-labeling
```

### Notebook Execution Order

```bash
# 1. EDA (~5 menit, CPU)
jupyter notebook notebooks/01_EDA.ipynb

# 2. Label cleaning (~10 menit, CPU)
# Jalankan semua cell, buka outputs/label_review.xlsx, isi kolom label_fix
jupyter notebook notebooks/02_LabelCleaning.ipynb

# 3. Baseline (~30-60 menit, CPU)
jupyter notebook notebooks/03_Baseline.ipynb

# 4. BERT training (~2-3 jam, GPU T4)
# Checkpoint disimpan per fold — aman jika session crash
jupyter notebook notebooks/04_AdvancedBERT.ipynb

# 5. Optuna tuning (~2 jam, GPU T4)
jupyter notebook notebooks/05_Optuna.ipynb
```

### Kaggle Setup

```python
# Di Kaggle Notebook, tambahkan cell pertama:
import sys, os
sys.path.insert(0, '/kaggle/input/mbg-case1-src')  # upload src/ sebagai dataset

cfg = Config(
    output_dir    = '/kaggle/working/outputs',
    labeled_file  = '/kaggle/input/mbg-dataset/hasil_label_baru.xlsx',
    test_file     = '/kaggle/input/mbg-dataset/case_1_text_to_predict.xlsx',
    template_file = '/kaggle/input/mbg-dataset/case_1_template_sheet.xlsx',
)
```

### Checkpoint Resume

Jika session Kaggle/Colab crash di tengah training, cukup jalankan cell yang sama lagi:

```python
# run_cv() otomatis skip fold yang sudah selesai
# State disimpan di:
# outputs/checkpoints/xlmr_fold1.pt       ← model weights
# outputs/checkpoints/xlmr_oof_logits.npy ← OOF logits terakumulasi
# outputs/checkpoints/xlmr_done_folds.txt ← list fold yang sudah selesai
```

---

## Key Findings & Next Steps

### Temuan Utama

1. **XLM-R jauh lebih kuat dari IndoBERT** (+10.3 poin OOF). Konsisten dengan literatur — XLM-R dilatih pada korpus jauh lebih besar meskipun bukan monolingual Indonesian.

2. **SMOTE tidak efektif untuk TF-IDF sparse vectors**. Interpolasi di feature space 50k-dimensional tidak menghasilkan synthetic samples yang semantically valid. `class_weight='balanced'` lebih efektif dan 10× lebih cepat.

3. **Label noise 25.2% adalah masalah nyata**. Politik (31.3%) dan Lainnya (31.7%) adalah dua kelas paling noisy — sesuai dengan performa terendah keduanya di BERT ensemble (0.672 dan 0.695).

4. **Ensemble sederhana memberikan +1.0 poin** (0.7308 → 0.7408) dengan bobot optimal w_xlmr=0.6. Gratis setelah kedua model sudah di-train.

5. **Kelas Ekonomi (145 → 196 samples setelah LLM re-label)** performanya 0.776 — lebih baik dari yang diperkirakan. WeightedRandomSampler + FocalLoss bekerja efektif untuk minority class ini.

## Dependencies

```
torch>=2.1.0
transformers>=4.38.0
accelerate>=0.26.0
optuna>=3.6.0
scikit-learn>=1.4.0
imbalanced-learn>=0.12.0
cleanlab>=2.6.0
pandas>=2.1.0
numpy>=1.26.0
openpyxl>=3.1.0
lightgbm>=4.3.0
```

---

*Last updated: May 2026 · BDC Internal Competition Case 1*
