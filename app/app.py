import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================================
# Konfigurasi halaman Streamlit
# ======================================================================

st.set_page_config(
    page_title="Analisis Sentimen Vibecoding",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# Konfigurasi path proyek dan resource
# ======================================================================

try:
    from config.setting import DATA_DIR, MODEL_DIR

    DATA_DIR = Path(DATA_DIR)
    MODEL_DIR = Path(MODEL_DIR)

except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / "models"

DEFAULT_DATA_PATH = DATA_DIR / "vibe_coding_auditLabel.csv"

# Model utama (hasil fine-tuning terbaik)
FINE_TUNED_MODEL_DIR = (
    MODEL_DIR
    / "fineTuneIndobert"
    / "oversample_only_e4_lr2e-05_tar250_ml256"
)

# ======================================================================
# Fungsi utilitas
# ======================================================================

@st.cache_data
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")

    df = pd.read_csv(path, sep=";")
    return df


@st.cache_resource
def load_model_tokenizer(path: Path):
    """
    Memuat tokenizer dan model dari direktori tertentu.
    Jika gagal, mengembalikan (None, error_message).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path))
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        return tokenizer, model
    except Exception as e:
        return None, str(e)

# ======================================================================
# Styling CSS
# ======================================================================

CUSTOM_CSS = """
<style>
    .main-title {
        color: #1f77b4;
        margin-bottom: 5px;
    }
    .subtitle {
        color: #555555;
        margin-bottom: 20px;
    }
    .header-style {
        color: #2c3e50;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    .subheader-style {
        color: #34495e;
        padding-left: 10px;
        border-left: 4px solid #3498db;
        margin-top: 15px;
        margin-bottom: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ======================================================================
# Load Dataset
# ======================================================================

try:
    df = load_dataset(DEFAULT_DATA_PATH)
except Exception as e:
    st.error(f"Gagal memuat dataset: {e}")
    st.stop()

total_komentar = len(df)
pos_count = len(df[df["sentiment_pseudo"] == "positif"])
net_count = len(df[df["sentiment_pseudo"] == "netral"])
neg_count = len(df[df["sentiment_pseudo"] == "negatif"])

# ======================================================================
# Sidebar Navigation
# ======================================================================

with st.sidebar:
    st.subheader("Ringkasan Dataset")
    st.markdown("---")
    st.text(f"Total komentar: {total_komentar}")
    st.text(f"Sentimen positif: {pos_count}")
    st.text(f"Sentimen netral:  {net_count}")
    st.text(f"Sentimen negatif: {neg_count}")
    st.markdown("---")

    menu = st.radio(
        "Pilih menu:",
        [
            "Beranda",
            "Analisis Sentimen",
            "Statistik",
            "Prediksi Teks",
            "Tentang",
        ],
    )

# ======================================================================
# Menu Beranda
# ======================================================================

def show_home():
    st.markdown(
        "<h1 class='main-title'>Analisis Sentimen Fenomena Vibecoding</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p class='subtitle'>Aplikasi ini menggunakan IndoBERT yang telah di-fine-tune untuk "
        "menganalisis sentimen komentar YouTube terkait fenomena Vibecoding.</p>",
        unsafe_allow_html=True,
    )

    # Informasi umum
    st.markdown("<div class='header-style'><h3>Informasi Umum</h3></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(
            """
            Proyek ini bertujuan untuk memahami persepsi publik terhadap fenomena Vibecoding
            dengan menganalisis komentar-komentar di YouTube menggunakan model bahasa Indonesia 
            berbasis IndoBERT.
            """
        )
        st.write(
            """
            Tahapan proses:
            1. Scraping komentar YouTube.
            2. Pembersihan dan normalisasi teks.
            3. Pelabelan sentimen (pseudo-label dan manual).
            4. Fine-tuning IndoBERT.
            5. Evaluasi dan deployment melalui Streamlit.
            """
        )

    with col2:
        st.write("Ringkasan Dataset")
        st.write(f"Total komentar: {total_komentar}")
        st.write(f"Positif: {pos_count}")
        st.write(f"Netral: {net_count}")
        st.write(f"Negatif: {neg_count}")

    # Hasil pelatihan model
    st.markdown("<div class='header-style'><h3>Hasil Evaluasi Model</h3></div>", unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Accuracy", "0.6875")
        st.metric("Eval Loss", "0.8879")
    with m2:
        st.metric("F1 Macro", "0.5873")
        st.metric("F1 Weighted", "0.7055")
    with m3:
        st.metric("Epochs", "4")
        st.metric("Learning Rate", "2e-05")

    st.markdown("<div class='subheader-style'><h4>Konfigurasi Model</h4></div>", unsafe_allow_html=True)
    config_df = pd.DataFrame(
        {
            "Parameter": ["epochs", "learning_rate", "batch_size", "target_per_class", "max_length"],
            "Nilai": [4, "2e-05", 8, 250, 256],
        }
    )
    st.table(config_df)

    # Struktur proyek
    st.markdown("<div class='header-style'><h3>Struktur Proyek</h3></div>", unsafe_allow_html=True)
    st.write(
        """
        - app/: aplikasi Streamlit
        - models/: model IndoBERT dan hasil fine-tuning
        - data/: dataset utama dan hasil praproses
        - notebooks/: eksperimen dan pelatihan model
        - src/: modul scraping dan sanitasi teks
        """
    )

# ======================================================================
# Menu Analisis Dataset
# ======================================================================

def show_analysis(df):
    st.markdown("<div class='header-style'><h2>Analisis Sentimen Berdasarkan Dataset</h2></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown("<div class='subheader-style'><h4>Distribusi Sentimen</h4></div>", unsafe_allow_html=True)
        label_counts = df["sentiment_pseudo"].value_counts().rename("jumlah")
        st.dataframe(label_counts)

    with col2:
        st.markdown("<div class='subheader-style'><h4>Visualisasi Distribusi</h4></div>", unsafe_allow_html=True)
        st.bar_chart(label_counts)

    st.markdown("<div class='subheader-style'><h4>Contoh Data</h4></div>", unsafe_allow_html=True)
    st.dataframe(df[["author", "text_raw", "sentiment_pseudo"]].head(20))

# ======================================================================
# Menu Statistik Tambahan
# ======================================================================

def show_stats(df):
    st.markdown("<div class='header-style'><h2>Statistik Tambahan</h2></div>", unsafe_allow_html=True)

    if "published_at" in df.columns:
        df_time = df.copy()
        df_time["published_at"] = pd.to_datetime(df_time["published_at"], errors="coerce")
        df_time = df_time.dropna(subset=["published_at"])
        df_time["date"] = df_time["published_at"].dt.date

        st.markdown("<div class='subheader-style'><h4>Tren Komentar per Tanggal</h4></div>", unsafe_allow_html=True)
        daily_counts = df_time.groupby("date").size()
        st.line_chart(daily_counts)

    if "text_trunc" in df.columns:
        st.markdown("<div class='subheader-style'><h4>Distribusi Panjang Teks</h4></div>", unsafe_allow_html=True)
        df_len = df.copy()
        df_len["text_length"] = df_len["text_trunc"].astype(str).str.len()
        st.write(df_len["text_length"].describe())
        st.bar_chart(df_len["text_length"].value_counts().sort_index())

# ======================================================================
# Menu Prediksi Teks (Model)
# ======================================================================

def show_prediction():
    st.markdown("<div class='header-style'><h2>Prediksi Sentimen Teks</h2></div>", unsafe_allow_html=True)
    st.write("Masukkan teks komentar untuk memprediksi sentimennya menggunakan model IndoBERT.")

    st.write(f"Model yang digunakan: {FINE_TUNED_MODEL_DIR}")

    if not FINE_TUNED_MODEL_DIR.exists():
        st.error("Folder model tidak ditemukan. Pastikan model telah ditempatkan pada direktori yang benar.")
        return

    user_text = st.text_area("Teks komentar", height=150)

    if st.button("Analisis Sentimen", use_container_width=True):
        if not user_text.strip():
            st.warning("Masukkan teks terlebih dahulu.")
            return

        tokenizer, model_or_error = load_model_tokenizer(FINE_TUNED_MODEL_DIR)

        if tokenizer is None:
            st.error(f"Gagal memuat model: {model_or_error}")
            return

        model = model_or_error

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        enc = tokenizer(
            user_text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]

        label_map = {0: "Negatif", 1: "Netral", 2: "Positif"}
        pred_label = label_map.get(pred_idx, "Tidak diketahui")

        col1, col2 = st.columns([2, 3])

        with col1:
            st.metric("Prediksi Sentimen", pred_label)
            st.metric("Confidence", f"{confidence*100:.1f}%")
            st.caption(f"Menggunakan perangkat: {'GPU' if torch.cuda.is_available() else 'CPU'}")

        with col2:
            df_prob = pd.DataFrame(
                {"Sentimen": [label_map[i] for i in range(len(probs))], "Probabilitas": probs}
            ).set_index("Sentimen")
            st.bar_chart(df_prob)

# ======================================================================
# Menu Tentang
# ======================================================================

def show_about():
    st.markdown("<div class='header-style'><h2>Tentang Aplikasi</h2></div>", unsafe_allow_html=True)
    st.write(
        """
        Aplikasi ini merupakan platform analisis sentimen berbasis model IndoBERT 
        yang telah di-fine-tune pada dataset komentar YouTube terkait Vibecoding.
        """
    )
    st.write(
        """
        Teknologi yang digunakan:
        - Streamlit
        - HuggingFace Transformers (IndoBERT)
        - PyTorch
        - Pandas dan NumPy
        """
    )

# ======================================================================
# Routing menu
# ======================================================================

if menu == "Beranda":
    show_home()
elif menu == "Analisis Sentimen":
    show_analysis(df)
elif menu == "Statistik":
    show_stats(df)
elif menu == "Prediksi Teks":
    show_prediction()
elif menu == "Tentang":
    show_about()