"""
testGroqKeys.py
---------------
Test script untuk verify Groq API keys configuration.

Usage:
    python models/testGroqKeys.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.GroqKeyManager import load_groq_env, collect_groq_keys

ENV_PATH = Path(__file__).resolve().parents[1] / "envs" / ".envs"

if __name__ == "__main__":
    print("Loading Groq API keys from envs/.envs...")
    load_groq_env(ENV_PATH)
    
    keys = collect_groq_keys()
    print(f'\nFound {len(keys)} API keys:')
    for i, k in enumerate(keys):
        print(f'  [{i}]: {k[:40]}...')
    
    if not keys:
        print("[WARNING] Tidak ada API keys ditemukan!")
    else:
        print(f"\n✓ API keys berhasil di-load. Siap untuk digunakan di AutoLabelSentiment.py")
