"""
AutoLabelSentiment.py
----------------------
Generate Pseudo-label sentiment (netral, positif, negatif)
untuk komentar fenomena "vibe coding" menggunakan Groq + Gemme2-9b-it

Input : data/vibe_coding_dataset_ready.csv
    Dataset harus punya kolom 'text_raw' dari Detokenization.py
Output : data/vibe_coding_with_pseudolables.csv
    - sentiment_pseudo          (str: "negatif"/"netral"/"positif")
    - sentiment_id_pseudo       (int: 0/1/2)

Usage:
    python model/AutoLabelSentiment.py
    python -m projek-analisis-sentimen-fenomena-vibecoding.models.AutoLabelSentiment
"""

import os
import time
import argparse
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from groq import Groq
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.setting import ENV_PATH, PROJECT_ROOT

# Handle both dev/ and models/ paths for GroqKeyManager
try:
    from dev.GroqKeyManager import load_groq_env, collect_groq_keys
except ModuleNotFoundError:
    # If running from dev/, adjust path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
    from GroqKeyManager import load_groq_env, collect_groq_keys


# ================================================================
# Configurasi dasar model
# ================================================================

MODEL_LIST = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
]
# Model rotation strategy: Rotate every 300-350 processed items
# Purpose: Distribute model load evenly to avoid hitting per-model rate limits
# Rotation interval: 325 items (middle of 300-350 range) per model
CURRENT_MODEL_INDEX: int = 0
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 3.5  # detik, untuk stay di bawah 30 requests/minute limit (17 req/min, very safe)

# ================================================================
# Prompt and Labelling
# ================================================================

def build_prompt(text:str) -> str:
    """
    Bangung prompt few-shot untuk klasifikasi sentimen 3 kelas
    Fokus: sikap terhadap fenomena "vibe coding".
    """

    guideline="""
Kamu adalah analis sentimen bahasa indonesia.

Tugasmu: Klasifikasikan komentar tentang fenomena "vibe coding"
ke dalam salah satu dari 3 label:
- negatif
- positif
- netral

definisi:
- postif: sikap mendukung / antusias / melihat dampak baik dari vibe coding
    (misalnya: memudahkan belajar, bikin semangat, membantu produktivitas)
- negatif: sikap menolak / khawatir / menialai vibe coding berdampak buruk
    (misalnya: berbahaya untuk pemula, menurunkan kualitas programmer, ancaman kerja)
- netral: bertanya / mengamati / sharing tanpa sikap jelas atau komentar yang tidak relevan lansung ke fenomena vibe coding.

Jika komentar campuran (ada plus dan minus):
- Kalau jelas dominan positif, pilih positif
- Kalau jelas dominan negattif, pilih negatif
- Kalau benar-benar seimbang atau ambigu, pilih netral

Output HANYA salah satu kata berikut (lowercase, tanpa tanda kutip):
negatif
positif
netral
""".strip()

    example_block="""
Contoh:
Komentar: "gue jadi makin bersemangat ngoding sejak tau vibe coding gini, ngebantu banget"
Label: positif

Komentar: "cara kayak gini cocok banget buat orang visual, jadi kebayang alurnya"
Label: positif

Komentar: "ini bahaya banget buat pemula, jadi gak belajar konsep dasar sama sekali"
Label: negatif

Komentar: "kalau semua ngandelin vibe coding kayak gini, kualitas programmer bisa turun jauh"
Label: negatif

Komentar: "bedanya vibe coding sama pake copilot biasa itu apa ya?"
Label: netral

Komentar: "di kantor gue belum ada yang pake vibe coding, masih manual aja"
Label: netral
""".strip()

    query_block=f"""
Sekarang klasifikasikan komentar iniL

Komentar: "{text}"

Jawab hanya dengan salah satu:
negatif
positif
netral

Label: 
""".strip()

    full_prompt = guideline + "\n\n" + example_block + "\n\n" + query_block
    return full_prompt

def normalize_label(raw_output:str) -> Literal["negatif", "netral", "positif"]:
    """
    Normalisasi output model ke label yang valid.
    Hanya menerima "negatif", "netral", "positif" sebagai output.
    """
    text = (raw_output or "").strip().lower()

    # Kalau model kepanjangan (kalimat), cari kata yang valid saja
    if "negatif" in text and "positif" in text:
        idx_neg = text.find("negatif")
        idx_net = text.find("netral") if "netral" in text else 10**9
        idx_pos = text.find("positif")
        first_idx = min(idx_neg, idx_net, idx_pos)
        if first_idx == idx_neg:
            return "negatif"
        elif first_idx == idx_net:
            return "netral"
        else:
            return "positif"
    
    if "negatif" in text:
        return "negatif"
    if "netral" in text:
        return "netral"
    if "positif" in text:
        return "positif"
    
    # fallback nama mirip
    if "negat" in text:
        return "negatif"
    if "netr" in text:
        return "netral"
    if "posit" in text:
        return "positif"

    # kalau benar-benar 'rancu' / ga jelas -> netral
    return "netral"

def sentiment_to_id(label:str) -> int:
    mappin={
        "negatif": 0,
        "netral": 1,
        "positif": 2,
    }
    return mappin.get(label, 1) # default netral

GROQ_KEYS: list[str] = []
CURRENT_KEY_INDEX: int = 0
CLIENT: Groq | None = None
FAILED_KEYS: set[int] = set()  # Track indices of API keys that have failed

def _collect_groq_keys() -> list[str]:
    """Wrapper untuk collect_groq_keys dari GroqKeyManager."""
    return collect_groq_keys()

def _build_client(idx: int) -> Groq:
    return Groq(api_key=GROQ_KEYS[idx])

def _rotate_client() -> bool:
    global CURRENT_KEY_INDEX, CLIENT, FAILED_KEYS
    
    # Try mencari API key yang belum gagal
    for idx in range(len(GROQ_KEYS)):
        if idx not in FAILED_KEYS:
            if idx != CURRENT_KEY_INDEX:
                CURRENT_KEY_INDEX = idx
                CLIENT = _build_client(CURRENT_KEY_INDEX)
                print(f"[INFO] Switch ke GROQ key index: {CURRENT_KEY_INDEX}")
                return True
            # Kalau sekarang lagi pakai key yang ok, coba continue
            return True
    
    # Kalau semua API key sudah gagal
    print("[ERROR] Semua API keys sudah mencapai rate limit!")
    return False

def _is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    code = getattr(e, "status_code", None) or getattr(e, "http_status", None) or getattr(e, "code", None)
    if isinstance(code, int) and code in (429, 403):
        return True
    signals = [
        "rate limit",
        "quota",
        "too many requests",
        "exceeded",
    ]
    return any(s in msg for s in signals)

def _is_auth_error(e: Exception) -> bool:
    msg = str(e).lower()
    code = getattr(e, "status_code", None) or getattr(e, "http_status", None) or getattr(e, "code", None)
    if isinstance(code, int) and code in (401,):
        return True
    return ("invalid api key" in msg) or ("authentication" in msg) or ("unauthorized" in msg)

def _print_key_status() -> None:
    """Print status dari semua API keys (active dan failed)."""
    active_keys = [i for i in range(len(GROQ_KEYS)) if i not in FAILED_KEYS]
    failed_keys = list(FAILED_KEYS)
    
    print(f"\n[STATUS] API Keys Summary:")
    print(f"  Current Active Key: index {CURRENT_KEY_INDEX}")
    print(f"  Available Keys: {active_keys}")
    print(f"  Failed Keys: {failed_keys}")
    print(f"  Total Available: {len(active_keys)}/{len(GROQ_KEYS)}\n")

def call_model(text:str) -> Literal["negatif", "netral", "positif"]:
    global CURRENT_KEY_INDEX, FAILED_KEYS
    prompt = build_prompt(text)
    max_retries = len(GROQ_KEYS)
    retry_count = 0
    start_time = time.time()
    
    while retry_count < max_retries:
        try:
            request_start = time.time()
            chat_completion = CLIENT.chat.completions.create(
                model=MODEL_LIST[CURRENT_MODEL_INDEX],
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah analis sentimen yang sangat disiplin dan hanya mnejawab label.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=10
            )
            request_elapsed = time.time() - request_start
            
            # Reset failed keys jika success (API key berhasil digunakan lagi)
            if CURRENT_KEY_INDEX in FAILED_KEYS:
                FAILED_KEYS.discard(CURRENT_KEY_INDEX)
                print(f"[INFO] API key index {CURRENT_KEY_INDEX} berhasil digunakan kembali")
            
            raw_output = chat_completion.choices[0].message.content
            label = normalize_label(raw_output)
            
            # Debug: Print timing info setiap 10 requests
            total_elapsed = time.time() - start_time
            if retry_count % 10 == 0:
                print(f"  [DEBUG] Key[{CURRENT_KEY_INDEX}] Response time: {request_elapsed:.2f}s, Total: {total_elapsed:.2f}s")
            
            return label
        except Exception as e:
            error_msg = str(e)
            request_elapsed = time.time() - request_start
            
            print(f"[WARN] Error pada key index {CURRENT_KEY_INDEX} (attempt {retry_count + 1}): {error_msg[:80]}...")
            print(f"       Response time: {request_elapsed:.2f}s")
            
            # Mark current key as failed jika rate limit atau auth error
            if _is_rate_limit_error(e):
                FAILED_KEYS.add(CURRENT_KEY_INDEX)
                _print_key_status()
                print(f"[WARN] API key index {CURRENT_KEY_INDEX} kena rate limit, ditandai sebagai gagal")
            elif _is_auth_error(e):
                FAILED_KEYS.add(CURRENT_KEY_INDEX)
                _print_key_status()
                print(f"[WARN] API key index {CURRENT_KEY_INDEX} auth error, ditandai sebagai gagal")
            
            # Coba rotate ke API key lain
            if _rotate_client():
                retry_count += 1
                print(f"[INFO] Mencoba dengan API key yang berbeda...")
                time.sleep(1)  # Wait sebentar sebelum retry
                continue
            else:
                # Semua API keys sudah gagal
                _print_key_status()
                print(f"[ERROR] Tidak ada API key yang tersedia. Return netral as fallback.")
                return "netral" # type: ignore[return-value]
    
    # Fallback jika max retries tercapai
    print(f"[ERROR] Max retries ({max_retries}) tercapai. Return netral as fallback.")
    return "netral" # type: ignore[return-value]

# ================================================================
# Main Script
# ================================================================

def main():
    load_groq_env(ENV_PATH)

    parser = argparse.ArgumentParser(description="Auto-label Sentiment komentar vibe coding (Groq)")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROJECT_ROOT / "data" / "vibe_coding_dataset_ready.csv"),
        help="Path ke file CSV input, harus ada kolom 'text_raw' (default: data/vibe_coding_dataset_ready.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "vibe_coding_with_pseudolabels.csv"),
        help="Path ke csv output dengan Pseudo-label"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Index baris mulai diproses (buat resume ketika script kepotong)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Index baris terakhir (exclusive). -1 = sampai habis.",
    )
    args = parser.parse_args()
    in_raw = Path(args.input)
    out_raw = Path(args.output)
    in_candidates = []
    if in_raw.parts and in_raw.parts[0] == PROJECT_ROOT.name:
        in_candidates.append(PROJECT_ROOT / Path(*in_raw.parts[1:]))
    in_candidates.extend([in_raw, PROJECT_ROOT / in_raw])
    in_path = None
    for cand in in_candidates:
        if cand.exists():
            in_path = cand
            break
    if in_path is None:
        raise FileNotFoundError(f"Input file {in_raw} tidak ditemukan relatif ke {PROJECT_ROOT}")
    if out_raw.is_absolute():
        out_path = out_raw
    else:
        if out_raw.parts and out_raw.parts[0] == PROJECT_ROOT.name:
            out_path = PROJECT_ROOT / Path(*out_raw.parts[1:])
        else:
            out_path = PROJECT_ROOT / out_raw

    global GROQ_KEYS, CURRENT_KEY_INDEX, CLIENT, FAILED_KEYS, CURRENT_MODEL_INDEX
    GROQ_KEYS = _collect_groq_keys()
    if not GROQ_KEYS:
        raise ValueError("API_KEY_GROQ tidak ditemukan. Tambahkan API_KEY_GROQ atau GROQ_API_KEY di envs/.envs")
    CURRENT_KEY_INDEX = 0
    CLIENT = _build_client(CURRENT_KEY_INDEX)
    FAILED_KEYS = set()  # Reset failed keys tracker
    print(f"[INFO] Loaded {len(GROQ_KEYS)} API keys untuk Groq")
    _print_key_status()

    if not in_path.exists():
        raise FileNotFoundError(f"Input file {in_path} tidak ditemukan.")

    df = pd.read_csv(in_path)

    if "text_raw" not in df.columns:
        raise FileNotFoundError("Kolom 'text_raw' tidak ditemukan. Pastikan sudah menjalankan model/Detokenization.py")
    
    n = len(df)
    start = max(0, args.start)
    end = n if args.end < 0 else min(args.end, n)

    print(f"Total rows      :   {n}")
    print(f"Processing      :   [{start}:{end}] (Total {end- start})")
    print(f"Available Models : {MODEL_LIST}")
    print(f"Current Model idx: {CURRENT_MODEL_INDEX} ({MODEL_LIST[CURRENT_MODEL_INDEX]})")

    sentiments = []
    sentiments_ids = []
    start_time = time.time()

    for idx in range(start, end):
        row = df.iloc[idx]
        text = str(row['text_raw']).strip()
        # Rotate model every 300-350 processed items (random between 300-350)
        rel_idx = idx - start + 1
        # Use 325 as middle point for rotation (300-350 range)
        model_idx = ((rel_idx - 1) // 325) % len(MODEL_LIST)
        if model_idx != CURRENT_MODEL_INDEX:
            CURRENT_MODEL_INDEX = model_idx
            print(f"\n[INFO] Rotate model -> index {CURRENT_MODEL_INDEX}: {MODEL_LIST[CURRENT_MODEL_INDEX]}\n")

        if not text:
            label = "netral"
        else:
            label = call_model(text)
        label_id = sentiment_to_id(label)
        
        sentiments.append(label)
        sentiments_ids.append(label_id)

        if rel_idx % 20 == 0 or idx == end - 1:
            elapsed = time.time() - start_time
            rate = (rel_idx / max(elapsed, 0.1)) * 60  # requests per minute
            print(f"Processed {rel_idx} / {end - start} (Rate: {rate:.1f} req/min, Elapsed: {elapsed/60:.1f}min)")
        
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Copy df dan tulis hasil ke range [start:end]
    df_out = df.copy()
    df_out.loc[start:end - 1, "sentiment_pseudo"] = sentiments
    df_out.loc[start:end - 1, "sentiment_pseudo_id"] = sentiments_ids

    os.makedirs(out_path.parent, exist_ok=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8")

    print("Selesai pseudo-labelling")
    print(f"Output disimpan ke {out_path}")

if __name__ == "__main__":
    main()
