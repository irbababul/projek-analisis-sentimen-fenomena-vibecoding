# Rate Limit Debugging Guide

## Masalah yang Anda Hadapi

```
groqRateLimitCheck: OK dengan 2.5s delay
AutoLabelSentiment: Masih kena rate limit (429 error)
```

**Mengapa bisa berbeda?**

### Perbedaan Test vs Production

| Aspek               | groqRateLimitCheck  | AutoLabelSentiment               |
| ------------------- | ------------------- | -------------------------------- |
| **Jumlah requests** | 5-10 requests saja  | 100+ requests                    |
| **Durasi**          | ~30 detik           | Berjam-jam                       |
| **Prompt size**     | Kecil (test prompt) | Besar (full guidance + examples) |
| **Response time**   | Consistent          | Bisa vary                        |
| **Error handling**  | Minimal             | Ada retries                      |

### Root Causes yang Mungkin

1. **Network Latency**

   ```
   SLEEP_BETWEEN_CALLS = 2.5s dimulai SETELAH response diterima

   Jika actual response time = 0.5s:
     Total = 2.5s + 0.5s = 3.0s per request
     Rate = 20 req/min ✓ AMAN

   Tapi jika response time = 1.5s (network slow):
     Total = 2.5s + 1.5s = 4.0s per request
     Rate = 15 req/min ✓ AMAN

   Atau jika ada spike:
     Total = 2.5s + 2.0s = 4.5s per request
     Tapi beberapa requests bisa "stack up" jika timing unlucky
   ```

2. **Prompt Size Impact**

   ```
   AutoLabelSentiment prompt:
   - System message: ~100 chars
   - Guidelines: ~800 chars
   - Examples: ~600 chars
   - Query: ~200 chars
   Total: ~1700 chars per request

   Groq perlu waktu lebih untuk parse dan respond
   ```

3. **Timing Precision**
   ```
   Script mengirim request setiap 2.5s
   Tapi jika requests "stack up" dalam Groq's queue,
   bisa exceed per-minute limit meskipun interval ok
   ```

---

## Solusi yang Sudah Diterapkan

### 1. ✅ Increase SLEEP_BETWEEN_CALLS

```python
SLEEP_BETWEEN_CALLS = 3.5  # dari 2.5
# Ini = 17 req/min, jauh di bawah limit 30 req/min
# Margin: 43% safety buffer
```

### 2. ✅ Detailed Logging

```
[DEBUG] Response time: 0.45s
[DEBUG] Key[0] Response time: 0.52s
...
```

Output ini membantu Anda lihat actual rate vs expected.

### 3. ✅ Rate Calculation in Output

```
Processed 20 / 100 (Rate: 18.5 req/min, Elapsed: 1.8min)
```

Ini show actual rate yang sedang berjalan.

---

## Bagaimana Cara Debug Lebih Lanjut

### Step 1: Test dengan Small Batch

```bash
python models/AutoLabelSentiment.py \
  --input data/vibe_coding_dataset_ready.csv \
  --output data/test_output.csv \
  --start 0 --end 30
```

Lihat apakah masih kena limit atau tidak.

### Step 2: Monitor Output

```
[INFO] Loaded 2 API keys untuk Groq
[STATUS] API Keys Summary:
  Current Active Key: index 0
  Available Keys: [0, 1]
  Failed Keys: []
  Total Available: 2/2

Total rows      :   946
Processing      :   [0:30] (Total 30)
Model Groq      :   llama-3.1-8b-instant

  [DEBUG] Key[0] Response time: 0.45s, Total: 0.23s
  [DEBUG] Key[0] Response time: 0.52s, Total: 3.74s
Processed 20 / 30 (Rate: 18.5 req/min, Elapsed: 1.8min)
Processed 30 / 30 (Rate: 17.1 req/min, Elapsed: 2.9min)
```

### Step 3: Analyze

- ✓ Jika finish tanpa error 429 → OK
- ✗ Jika masih kena 429 → Increase SLEEP_BETWEEN_CALLS lebih lagi

### Step 4: Batch Processing (if needed)

```bash
# Proses dalam chunks untuk spread requests over time
python models/AutoLabelSentiment.py --start 0 --end 100
# Tunggu 5 menit

python models/AutoLabelSentiment.py --start 100 --end 200
# Tunggu 5 menit

python models/AutoLabelSentiment.py --start 200 --end 300
# Dst...
```

---

## Recommended Settings

### Conservative (Very Safe) - Recommended

```python
SLEEP_BETWEEN_CALLS = 3.5  # 17 req/min
# Margin: 43% below limit
# Risk: Minimal, Processing time: Long
```

### Balanced

```python
SLEEP_BETWEEN_CALLS = 3.0  # 20 req/min
# Margin: 33% below limit
# Risk: Low, Processing time: Moderate
```

### Aggressive (Not Recommended)

```python
SLEEP_BETWEEN_CALLS = 2.0  # 30 req/min (AT LIMIT)
# Margin: 0%
# Risk: High, Processing time: Short
# ❌ Jangan pakai ini! Timing tidak konsisten di production
```

---

## Troubleshooting Checklist

- [ ] SLEEP_BETWEEN_CALLS = 3.5 (atau lebih)
- [ ] Run dengan --start 0 --end 30 dulu
- [ ] Monitor output untuk actual rate
- [ ] Jika masih error, increase ke 4.0
- [ ] Jika semua aman, bisa increase sedikit (3.5-4.0 adalah sweet spot)
- [ ] Untuk full dataset, gunakan batch processing

---

## Monitoring Tools

### Option 1: Visual Monitoring

```bash
# Jalankan script, lihat output live
python models/AutoLabelSentiment.py --start 0 --end 100
```

### Option 2: Silent Mode (Background)

```bash
# Run di background, check log nanti
python models/AutoLabelSentiment.py --start 0 --end 946 > output.log 2>&1 &
```

### Option 3: Batch Processing (Safest)

```bash
# Split ke chunks, process satu-satu
for chunk in 0-100 100-200 200-300 ...; do
  python models/AutoLabelSentiment.py --start $chunk
  sleep 300  # Wait 5 minutes
done
```
