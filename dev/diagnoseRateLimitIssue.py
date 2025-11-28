"""
diagnose_rate_limit_issue.py
----------------------------
Diagnose script untuk understand why AutoLabelSentiment kena limit
padahal groqRateLimitCheck ok.

Ini adalah simulation tanpa actual API calls!

Scenario Diagnosis:
1. Check timing: Apakah SLEEP_BETWEEN_CALLS dipatuhi?
2. Check concurrency: Apakah ada async/concurrent requests?
3. Check backoff: Apakah retry berjalan dengan benar?
4. Check rate calculation: Actual request rate vs expected
"""

import time
from datetime import datetime

def diagnose_rate_limit():
    print("=" * 70)
    print("DIAGNOSIS: Why AutoLabelSentiment hits rate limit")
    print("=" * 70)
    
    # Scenario 1: Check SLEEP_BETWEEN_CALLS impact
    print("\n[SCENARIO 1] Impact of SLEEP_BETWEEN_CALLS")
    print("-" * 70)
    
    scenarios = [
        ("0.5 detik", 0.5),
        ("2.5 detik", 2.5),
        ("3.0 detik", 3.0),
    ]
    
    groq_limit = 30  # requests per minute
    
    for name, sleep_time in scenarios:
        # Simulating 100 requests
        total_time = sleep_time * 100
        actual_rate = (100 / total_time) * 60 if total_time > 0 else 0
        
        status = "❌ TOO FAST" if actual_rate > groq_limit else "✓ SAFE"
        
        print(f"\n{name} delay:")
        print(f"  Total time for 100 requests: {total_time:.0f}s ({total_time/60:.1f}min)")
        print(f"  Actual rate: {actual_rate:.1f} req/min")
        print(f"  GROQ limit: {groq_limit} req/min")
        print(f"  Status: {status}")
    
    # Scenario 2: Potential hidden issues
    print("\n\n[SCENARIO 2] Potential Hidden Issues")
    print("-" * 70)
    
    issues = [
        ("Prompt size", 
         "Jika prompt terlalu besar, response lebih lambat\n"
         "  → Groq API lebih lama process\n"
         "  → Waktu yang diperkirakan tidak akurat\n"
         "  → Bisa kena timeout atau rate limit"),
        
        ("Network latency",
         "Response time bukan hanya server processing\n"
         "  → Ada latency jaringan\n"
         "  → SLEEP_BETWEEN_CALLS dimulai SETELAH request dikirim\n"
         "  → Jika response lambat, next request bisa jadi terlalu cepat"),
        
        ("Error retries",
         "Jika ada errors, akan retry dengan key lain\n"
         "  → Tapi tetap pakai SLEEP_BETWEEN_CALLS\n"
         "  → Jika retry terlalu sering, bisa accumulate"),
        
        ("Dashboard counting",
         "Dashboard Groq count successful requests\n"
         "  → Tapi rate limit adalah per-minute throttling\n"
         "  → Bisa ada pending/queued requests yang tidak tercatat"),
    ]
    
    for i, (title, explanation) in enumerate(issues, 1):
        print(f"\n{i}. {title}:")
        print(f"   {explanation.replace(chr(10), chr(10) + '   ')}")
    
    # Scenario 3: Recommended solution
    print("\n\n[SCENARIO 3] Recommended Solutions")
    print("-" * 70)
    
    solutions = [
        ("Increase SLEEP_BETWEEN_CALLS",
         "Dari 2.5s ke 3-4s untuk lebih aman",
         "SLEEP_BETWEEN_CALLS = 3.5  # 17 req/min, sangat aman"),
        
        ("Monitor actual response times",
         "Log waktu setiap request untuk debugging",
         "Sudah ditambahkan di call_model() function"),
        
        ("Batch processing dengan checkpoints",
         "Proses 50 text, save, tunggu, lanjut",
         "Bisa split dengan --start dan --end arguments"),
        
        ("Use different API keys strategically",
         "Jangan semua hit limit bersamaan",
         "Rotation system sudah ada, tapi timing bisa dioptimasi"),
    ]
    
    for i, (title, reason, example) in enumerate(solutions, 1):
        print(f"\n{i}. {title}")
        print(f"   Reason: {reason}")
        print(f"   Example: {example}")
    
    # Scenario 4: Test timing without burning quota
    print("\n\n[SCENARIO 4] Simulate Processing (No API calls)")
    print("-" * 70)
    
    print("\nSimulating 50 requests dengan SLEEP_BETWEEN_CALLS = 2.5s")
    simulated_start = time.time()
    
    num_requests = 50
    sleep_time = 2.5
    
    for i in range(num_requests):
        # Simulate request time (fake, 0.2-0.5s)
        request_time = 0.3
        
        if (i + 1) % 10 == 0:
            elapsed = time.time() - simulated_start
            rate = ((i + 1) / elapsed) * 60
            print(f"  [{i+1:2d}] Elapsed: {elapsed:6.1f}s, Rate: {rate:5.1f} req/min")
        
        time.sleep(sleep_time)
    
    elapsed = time.time() - simulated_start
    final_rate = (num_requests / elapsed) * 60
    
    print(f"\nFinal: {num_requests} requests in {elapsed/60:.1f}min = {final_rate:.1f} req/min")
    print(f"GROQ limit: 30 req/min")
    print(f"Status: {'✓ SAFE' if final_rate < 30 else '❌ TOO FAST'}")

def recommendation():
    print("\n\n" + "=" * 70)
    print("REKOMENDASI")
    print("=" * 70)
    
    rec = """
Berdasarkan diagnosis, ada beberapa kemungkinan:

1. SLEEP_BETWEEN_CALLS = 2.5s SEHARUSNYA AMAN
   - Tapi mungkin ada network latency yang tidak terhitung
   - Atau prompt processing lebih lambat dari expected

2. SOLUSI TERBAIK: Increase delay ke 3-4 detik
   SLEEP_BETWEEN_CALLS = 3.5  # 17 req/min, sangat konservatif

3. ATAU: Monitor actual rates dari logging yang sudah ditambahkan
   - Jalankan 50 requests
   - Lihat actual rate dari output
   - Adjust SLEEP_BETWEEN_CALLS berdasarkan hasil real

4. UNTUK TESTING TANPA BURN QUOTA:
   - Gunakan groqRateLimitCheck tapi dengan delay yang sesuai
   - Atau gunakan script ini untuk simulation
   - Jangan test dengan actual auto-labeling sampai yakin

5. BATCH PROCESSING STRATEGY:
   # Proses 100 text, save
   python models/AutoLabelSentiment.py --start 0 --end 100
   
   # Tunggu 10 menit, lanjut
   python models/AutoLabelSentiment.py --start 100 --end 200
   
   # Dst...
"""
    
    print(rec)

if __name__ == "__main__":
    diagnose_rate_limit()
    recommendation()
