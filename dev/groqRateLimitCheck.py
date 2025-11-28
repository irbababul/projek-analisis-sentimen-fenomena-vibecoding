"""
groq_rate_limit_check.py
------------------------
Script untuk check dan understand Groq rate limiting behavior.

Rate Limit Types:
1. Monthly quota (requests per month) - tidak reset
2. Hourly rate limit (requests per hour) - reset per jam
3. Minute rate limit (requests per minute) - reset per menit
4. Concurrent requests limit

Groq's typical limits:
- Free tier: 30 requests per minute, 9000 per day
- Paid tier: higher limits

Usage:
    python groq_rate_limit_check.py
"""

import time
from datetime import datetime
from groq import Groq
from pathlib import Path
import sys
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.GroqKeyManager import collect_groq_keys, load_groq_env

ENV_PATH = Path(__file__).resolve().parents[1] / "envs" / ".envs"

def test_rate_limits():
    load_groq_env(ENV_PATH)
    keys = collect_groq_keys()
    
    if not keys:
        print("[ERROR] Tidak ada API keys ditemukan")
        return
    
    print("=" * 70)
    print("GROQ RATE LIMIT TEST")
    print("=" * 70)
    print(f"\nTesting dengan {len(keys)} API key(s)\n")
    
    for key_idx, api_key in enumerate(keys):
        print(f"\n{'='*70}")
        print(f"Testing API Key [{key_idx}]: {api_key[:30]}...")
        print(f"{'='*70}")
        
        client = Groq(api_key=api_key)
        
        # Test 1: Single request
        print("\n[Test 1] Single Request Check")
        try:
            start = time.time()
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "user",
                        "content": "Respond dengan 'OK' saja"
                    }
                ],
                temperature=0.0,
                max_tokens=5
            )
            elapsed = time.time() - start
            
            print(f"  ✓ Request berhasil (took {elapsed:.2f}s)")
            print(f"  Response: {response.choices[0].message.content}")
            
            # Check headers untuk rate limit info
            if hasattr(response, 'headers'):
                print(f"  Headers: {response.headers}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_msg = str(e).lower()
            if "429" in str(e) or "rate limit" in error_msg:
                print(f"  → Rate limit detected!")
            elif "401" in str(e) or "invalid" in error_msg:
                print(f"  → Auth error (Invalid API key?)")
        
        # Test 2: Check request limits with small delay
        print("\n[Test 2] Sequential Requests (5 requests with 1s delay)")
        for i in range(5):
            try:
                start = time.time()
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": f"test {i}"}],
                    temperature=0.0,
                    max_tokens=3
                )
                elapsed = time.time() - start
                print(f"  Request {i+1}: ✓ ({elapsed:.2f}s)")
                time.sleep(1)
            except Exception as e:
                print(f"  Request {i+1}: ✗ Error - {str(e)[:60]}...")
                if "429" in str(e) or "rate limit" in str(e).lower():
                    print(f"  → Rate limit hit after {i+1} request(s)")
                break

def get_rate_limit_info():
    """
    Informasi tentang Groq rate limiting
    """
    print("\n" + "="*70)
    print("GROQ RATE LIMIT INFORMASI")
    print("="*70)
    
    info = """
FREE TIER GROQ API:
  • Monthly Quota: Unlimited* (tergantung total requests)
  • Rate Limit: 30 requests/minute
  • Daily Limit: ~9000 requests/day
  • Concurrent: 1 request at a time
  
*Unlimited dalam arti boleh banyak, tapi ada throttling per menit

PENYEBAB RATE LIMIT:
  1. Lebih dari 30 requests dalam 1 menit
  2. Terlalu banyak concurrent requests
  3. API key quota habis (monthly limit)
  4. Sistem Groq throttle karena excessive usage

SOLUSI:
  1. Tambah delay antar request (SLEEP_BETWEEN_CALLS)
  2. Jangan request secara concurrent
  3. Check dashboard untuk quota usage
  4. Gunakan multiple API keys dengan rotation
  5. Optimize prompt size (shorter = faster)

CATATAN PENTING:
  - Jika sudah pakai 2.2k calls, sisanya adalah monthly quota yang tersisa
  - Rate limit per menit ≠ quota per bulan
  - 2.2k calls ÷ ~30 days ≈ 73 calls/hari ≈ 3 calls/jam → OK
  - Tapi dalam 1 menit, harus ≤30 requests

STRATEGI OPTIMAL:
  1. SLEEP_BETWEEN_CALLS = 0.5 detik (sekarang)
     → ~2 requests/second = ~120 requests/minute ❌ TOO FAST
  
  2. SLEEP_BETWEEN_CALLS = 2 detik
     → 0.5 requests/second = 30 requests/minute ✓ AT LIMIT
  
  3. SLEEP_BETWEEN_CALLS = 2.5 detik
     → 0.4 requests/second = 24 requests/minute ✓ SAFE
"""
    
    print(info)

if __name__ == "__main__":
    test_rate_limits()
    get_rate_limit_info()
