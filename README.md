# 😊 Analisis Sentimen Fenomena Vibecoding

**Aplikasi machine learning untuk menganalisis sentimen komentar YouTube terkait fenomena Vibecoding** menggunakan model **IndoBERT** yang sudah di-fine-tune dengan performa F1-macro **0.5873**.

[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](https://github.com/RayhanLup1n/projek-analisis-sentimen-fenomena-vibecoding)
[![Branch](https://img.shields.io/badge/Branch-Dev%2FModelling-blue)](https://github.com/RayhanLup1n/projek-analisis-sentimen-fenomena-vibecoding)

---

## 🎯 Tujuan Proyek

Menganalisis **sentimen publik terhadap fenomena Vibecoding** melalui:

- Scraping komentar YouTube
- Pelabelan sentiment (Positif, Netral, Negatif)
- Fine-tuning model IndoBERT
- Deployment dengan UI interaktif (Streamlit)

---

## 🚀 Quick Start

### 1️⃣ Setup Environment

```bash
# Clone repository
git clone https://github.com/RayhanLup1n/projek-analisis-sentimen-fenomena-vibecoding.git
cd projek-analisis-sentimen-fenomena-vibecoding

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App

```bash
streamlit run app/app.py
```

Buka di browser: `http://localhost:8501`

### 3️⃣ Train Model (Opsional)

```bash
jupyter notebook notebooks/TrainBERT.ipynb
```

---

## 📁 Project Structure

```
projek-analisis-sentimen-fenomena-vibecoding/
│
├── 📂 app/                              # ⭐ Streamlit application
│   └── app.py                           # Main UI (5 menu pages)
│
├── 📂 models/                           # 🤖 Fine-tuned models
│   ├── fineTuneIndobert/                # Hyperparameter tuning results (6 configs)
│   │   └── oversample_only_e4_lr2e-05_tar250_ml256/  ⭐ BEST MODEL
│   └── experimentsIndobert/             # Baseline experiments (4 strategies)
│
├── 📂 data/                             # 📊 Dataset
│   ├── vibe_coding_auditLabel.csv       ⭐ Main dataset (879 labeled comments)
│   ├── vibe_coding_dataset_ready.csv
│   ├── vibe_coding_yt_comments.csv
│   ├── vibe_coding_yt_comments_clean.csv
│   └── vibe_coding_pseudoLabel.csv
│
├── 📂 notebooks/                        # 📓 Jupyter notebooks
│   ├── TrainBERT.ipynb                  ⭐ Main training (baseline + tuning)
│   └── eda.ipynb                        # Exploratory data analysis
│
├── 📂 src/                              # 🔧 Utility scripts
│   ├── Sanitizer.py                     # Text cleaning
│   └── Scrape.py                        # YouTube scraper
│
├── 📂 dev/                              # 🛠️ Development tools
│   ├── AutoLabelSentiment.py            # Auto-labeling with Groq API
│   ├── GroqKeyManager.py                # API key rotation
│   └── [other debugging tools]
│
├── 📂 config/                           # ⚙️ Configuration
│   └── setting.py                       # Global settings
│
├── 📂 docs/                             # 📖 Documentation
│   ├── TRAINING_GUIDE.md                # How to train models
│   ├── API_KEY_ROTATION.md
│   └── label_sentiment_guidelines.md
│
├── 📄 README.md                         ⭐ This file
├── 📄 FOLDER_TREE.txt                   # Visual tree structure
├── 📄 requirements.txt                  # Dependencies
└── [Other files & notebooks]
```

---

## 📊 Dataset Overview

| Aspek                     | Detail                                     |
| ------------------------- | ------------------------------------------ |
| **Total Comments**        | 879 komentar YouTube                       |
| **Labeling**              | Manual (audit label)                       |
| **Sentimen Distribution** | Negatif (15%), Netral (72%), Positif (13%) |
| **Train/Val Split**       | 80/20 stratified                           |
| **Preprocessing**         | Text cleaning, whitespace normalization    |

**Label Mapping**:

```python
0 = Negatif (134 samples)
1 = Netral  (630 samples)  # Majority class
2 = Positif (115 samples)
```

---

## 🤖 Model Architecture

### Best Model: `oversample_only_e4_lr2e-05_tar250_ml256`

**Configuration**:

- Base Model: `indolem/indobert-base-uncased` (Indonesian BERT)
- Strategy: Oversampling minority classes (tanpa class weights)
- Epochs: 4
- Learning Rate: 2e-5
- Batch Size: 8
- Max Length: 256 tokens
- Target per class: 250 (after oversampling)

**Performance Metrics**:

| Metric          | Value         |
| --------------- | ------------- |
| **F1-macro**    | **0.5873** ⭐ |
| **F1-weighted** | 0.7055        |
| **Accuracy**    | 0.6875        |
| **Eval Loss**   | 0.8879        |

### Training Summary

**4 Baseline Strategies**:

1. Baseline (no oversampling, no weights)
2. Class weight only
3. Oversampling only ← **Selected for tuning**
4. Oversampling + class weights

**6 Hyperparameter Tuning Configs**:

- `e3_lr2e-05_tar250_ml256` (F1: 0.5871)
- `e4_lr2e-05_tar250_ml256` **BEST** (F1: 0.5873)
- `e3_lr3e-05_tar250_ml256` (F1: 0.5666)
- `e4_lr3e-05_tar250_ml256` (F1: 0.4529)
- `e3_lr2e-05_tar300_ml256` (F1: 0.5598)
- `e3_lr2e-05_tar250_ml128` (F1: 0.5075)

---

## 🎨 Streamlit App Features

### Menu Pages:

1. **🏠 Beranda** (Home)

   - Deskripsi proyek
   - Informasi dataset & sentimen distribution
   - Technology stack overview

2. **📊 Analisis Sentimen** (Sentiment Analysis)

   - Upload CSV file
   - Preview data dengan format yang benar
   - Support format: `.csv` dengan delimiter `;`

3. **📈 Statistik** (Statistics)

   - Bar chart distribusi sentimen
   - Summary metrics (total, terbanyak, dll)
   - Visualisasi dengan matplotlib/plotly

4. **🔍 Prediksi Teks** (Text Prediction) ⭐ **Main Feature**

   - Input teks komentar
   - Model inference dengan caching
   - Output:
     - Predicted sentiment label
     - Confidence score (%)
     - Probability distribution (bar chart)
     - Keyword matching explanation
     - Tokenization preview
   - Model directory customizable

5. **⚙️ Tentang** (About)
   - Versi aplikasi
   - Team information
   - Contact & support

### Key Features:

- 🔧 **Model Caching**: Fast inference dengan `@st.cache_resource`
- 📊 **Visualizations**: Probability charts, keyword highlighting
- 🌍 **Bahasa**: Bilingual support (Bahasa Indonesia & English)
- 🎨 **Design**: Modern UI dengan custom CSS styling
- ⚡ **Performance**: CPU/GPU compatible

---

## 🔧 Technical Stack

| Komponen            | Technology                               |
| ------------------- | ---------------------------------------- |
| **ML Framework**    | PyTorch + HuggingFace Transformers       |
| **Model**           | IndoBERT (indolem/indobert-base-uncased) |
| **Training**        | HF Trainer dengan custom WeightedTrainer |
| **Data Processing** | pandas, numpy, scikit-learn, datasets    |
| **UI/Deployment**   | Streamlit                                |
| **Metrics**         | F1-macro, F1-weighted, Accuracy          |
| **Environment**     | Python 3.8+, CUDA optional               |

---

## 📖 Folder Details

### `app/` - Streamlit Application

- **File**: `app.py`
- **Purpose**: Main UI untuk inferensi & visualisasi
- **Features**: 5 menu pages, model caching, responsive design

### `models/` - Trained Models

- **fineTuneIndobert/**: Hasil hyperparameter tuning (6 configs)
- **experimentsIndobert/**: Baseline strategy experiments (4 configs)
- Setiap folder berisi: `config.json`, `model.safetensors`, `tokenizer.json`, `vocab.txt`, dll

### `data/` - Dataset

- **vibe_coding_auditLabel.csv**: Main labeled dataset (⭐ gunakan ini)
- **vibe_coding_dataset_ready.csv**: Preprocessed version
- **vibe_coding_yt_comments\*.csv**: Raw dan cleaned comments
- **vibe_coding_pseudoLabel.csv**: Auto-labeled dengan Groq API

### `notebooks/` - Jupyter Notebooks

- **TrainBERT.ipynb**: Complete training pipeline
  - Data loading & label mapping
  - Stratified train/val split
  - 4 baseline strategies
  - 6 hyperparameter tuning configs
  - Result comparison & best model selection
- **eda.ipynb**: Exploratory Data Analysis

### `src/` - Utility Scripts

- **Sanitizer.py**: Text cleaning (remove special chars, normalize)
- **Scrape.py**: YouTube comments scraper

### `dev/` - Development Tools

- **AutoLabelSentiment.py**: Auto-labeling dengan Groq API
- **GroqKeyManager.py**: API key rotation untuk rate limit handling
- **groqRateLimitCheck.py**: Monitor rate limit status
- Other debugging & testing utilities

### `config/` - Configuration

- **setting.py**: Global settings (PROJECT_ROOT, DATA_DIR, paths, dll)

### `docs/` - Documentation

- **TRAINING_GUIDE.md**: How to train, CLI examples, troubleshooting
- **API_KEY_ROTATION.md**: Setup API key rotation
- **label_sentiment_guidelines.md**: Sentiment labeling criteria---

## 🚀 How to Use

### 1. Run Streamlit App

```bash
streamlit run app/app.py
```

Then go to: `http://localhost:8501`

**Main workflow**:

1. Open **🔍 Prediksi Teks** menu
2. Input teks komentar (default model path will be used)
3. Click **🚀 Analisis Sentimen**
4. View results: sentiment label, confidence, probability chart, keyword explanation

### 2. Upload & Analyze Dataset

```bash
# In 📊 Analisis Sentimen menu
# Upload: vibe_coding_auditLabel.csv
# Preview data, check label distribution
```

### 3. Train New Model

```bash
jupyter notebook notebooks/TrainBERT.ipynb
```

**Steps in notebook**:

1. Load data & prepare splits (80/20)
2. Run 4 baseline strategies (4 experiments)
3. Run 6 hyperparameter tuning configs
4. Compare results & select best
5. Generate classification report

---

## 📊 Data Format

**CSV Structure** (`vibe_coding_auditLabel.csv`):

```
text_raw;sentiment_pseudo;[other columns...]
"Vibecoding itu keren banget!";positif;...
"Biasa aja sih";netral;...
"Gak suka deh";negatif;...
```

**Requirements**:

- Delimiter: `;` (semicolon)
- Encoding: UTF-8
- Sentiment values: "negatif", "netral", "positif" (lowercase)

---

## 🎯 Model Training Details

### Data Imbalance Solution

**Problem**: Netral class (630) >> Positif (115) & Negatif (134)

**Solution**: Oversampling minority classes to 250 samples per class

```python
Label distribution after oversampling:
- Negatif: 250 (sampled from 134)
- Netral: 630 (kept as is)
- Positif: 250 (sampled from 115)
```

### Training Process

1. **Data Loading** → `vibe_coding_auditLabel.csv`
2. **Label Mapping** → negatif/netral/positif → 0/1/2
3. **Train/Val Split** → 80/20 stratified
4. **Oversampling** → Minority classes → 250 samples
5. **Tokenization** → Max length 256, IndoBERT tokenizer
6. **Model** → IndoBERT + classification head (3 labels)
7. **Loss** → CrossEntropyLoss (with optional class weights)
8. **Metrics** → F1-macro, F1-weighted, Accuracy
9. **Evaluation** → On validation set
10. **Save** → Model + tokenizer to output directory

---

## ❓ FAQ

**Q: Bagaimana cara menggunakan model yang sudah dilatih?**

A: Cukup jalankan Streamlit app:

```bash
streamlit run app/app.py
```

Lalu gunakan menu **🔍 Prediksi Teks** untuk input teks & dapatkan prediksi.

**Q: Model disimpan di mana?**

A: `models/fineTuneIndobert/oversample_only_e4_lr2e-05_tar250_ml256/`

- Berisi: `model.safetensors`, `tokenizer.json`, `config.json`, dll

**Q: Bisakah saya melatih dengan dataset berbeda?**

A: Ya! Edit `notebooks/TrainBERT.ipynb` cell 3 untuk mengubah `data_path` ke file CSV Anda. Pastikan format sesuai (delimiter `;`, columns: `text_raw`, `sentiment_pseudo`).

**Q: Berapa lama proses training?**

A: Tergantung hardware:

- **GPU (NVIDIA A100)**: ~45 menit per eksperimen
- **CPU**: ~2-3 jam per eksperimen
- **Total (6 configs)**: 4.5-18 jam

**Q: Kenapa F1-macro dipilih sebagai metrik utama?**

A: Karena dataset **imbalanced**. F1-macro memberikan performa rata-rata per kelas, sehingga lebih adil untuk minority classes (Positif & Negatif).

**Q: Bagaimana cara meningkatkan akurasi model?**

A: Coba:

1. Tambah data labeling (lebih banyak samples)
2. Fine-tune learning rate (1e-5, 3e-5)
3. Adjust target_per_class saat oversampling
4. Coba class weights kombinasi
5. Gunakan model BERT yang lebih besar

**Q: Aplikasi crash saat predict?**

A: Pastikan:

1. Model path benar (default: `models/fineTuneIndobert/oversample_only_e4_lr2e-05_tar250_ml256/`)
2. Dependencies installed: `pip install -r requirements.txt`
3. GPU memory cukup (atau gunakan CPU)
4. Teks input tidak kosong

---

## 🔄 Model Selection Timeline

```
Initial Data (879 comments)
    ↓
Stratified Split (80/20)
    ↓
4 Baseline Strategies
├── baseline (no oversampling, no weights)
├── class_weight_only
├── oversample_only ⭐ SELECTED
└── oversample_and_weight
    ↓
6 Hyperparameter Tuning (oversample_only)
├── e3_lr2e-05_tar250_ml256 → F1: 0.5871
├── e4_lr2e-05_tar250_ml256 → F1: 0.5873 ⭐⭐ BEST
├── e3_lr3e-05_tar250_ml256 → F1: 0.5666
├── e4_lr3e-05_tar250_ml256 → F1: 0.4529
├── e3_lr2e-05_tar300_ml256 → F1: 0.5598
└── e3_lr2e-05_tar250_ml128 → F1: 0.5075
    ↓
✅ Final Model: oversample_only_e4_lr2e-05_tar250_ml256
```

---

## 🤝 Contributing & Development

### Setting Up Development Environment

```bash
# Clone & setup
git clone https://github.com/RayhanLup1n/projek-analisis-sentimen-fenomena-vibecoding.git
cd projek-analisis-sentimen-fenomena-vibecoding

# Create dev branch
git checkout -b feature/your-feature

# Install dev dependencies
pip install -r requirements.txt
pip install jupyter notebook

# Make changes
# ...

# Commit & push
git add .
git commit -m "Description of changes"
git push origin feature/your-feature
```

### Code Structure

- **Training logic**: `notebooks/TrainBERT.ipynb` (primary) or `models/TrainBERT.py` (CLI)
- **App logic**: `app/app.py`
- **Config**: `config/setting.py`
- **Utilities**: `src/` (data processing), `dev/` (debugging)

---

## 📞 Contact & Support

- **Repository**: https://github.com/RayhanLup1n/projek-analisis-sentimen-fenomena-vibecoding
- **Branch**: Dev/Modelling
- **Issues**: https://github.com/RayhanLup1n/projek-analisis-sentimen-fenomena-vibecoding/issues

Untuk pertanyaan, buka GitHub issue atau hubungi tim developers.

---

## 👥 Team

### Disusun Oleh:

1. **Rayhan Ananda Resky** (@RayhanLup1n)

   - Model architecture & training

2. **Muhammad Irbabul Salas**

   - Data scraping
   - Data Cleaning

3. **Muhammad Sawaludin**
   - Data annotation & labeling

---

## 📝 License & Repository Info

**Repository**: https://github.com/RayhanLup1n/projek-analisis-sentimen-fenomena-vibecoding
**Current Branch**: Dev/Modelling
**Default Branch**: main
**License**: MIT (or specify your license)

---

## 🎓 References & Resources

- [IndoBERT](https://huggingface.co/indolem/indobert-base-uncased) - Indonesian BERT Model
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) - ML Framework
- [Streamlit Documentation](https://docs.streamlit.io/) - App Framework
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) - Evaluation

---

**Last Updated**: November 28, 2025
**Version**: 1.0.0
**Status**: ✅ Production Ready
