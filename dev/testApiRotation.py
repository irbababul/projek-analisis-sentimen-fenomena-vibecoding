"""
test_api_rotation.py
--------------------
Test script untuk demo API key rotation ketika ada error.

Cara kerja:
1. Ubah salah satu API key di envs/.envs ke invalid key
2. Script akan otomatis switch ke API key yang valid
3. Lihat log status untuk melihat active dan failed keys

Usage:
    python test_api_rotation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.AutoLabelSentiment import main

if __name__ == "__main__":
    print("=" * 70)
    print("TEST API KEY ROTATION")
    print("=" * 70)
    print("\nMengetes auto-labeling dengan monitoring API key rotation...")
    print("Jika ada API key yang failed, script akan otomatis switch ke yang lain.\n")
    
    # Test dengan 10 baris untuk melihat behavior
    import argparse
    parser = argparse.ArgumentParser(description="Test API key rotation")
    parser.add_argument("--start", type=int, default=0, help="Start index")
    parser.add_argument("--end", type=int, default=10, help="End index")
    args = parser.parse_args()
    
    # Override sys.argv untuk main() bisa parse args
    sys.argv = [
        "test_api_rotation.py",
        "--input", "data/vibe_coding_dataset_ready.csv",
        "--output", "data/test_rotation_output.csv",
        "--start", str(args.start),
        "--end", str(args.end),
    ]
    
    main()
    
    print("\n" + "=" * 70)
    print("TEST SELESAI")
    print("=" * 70)
