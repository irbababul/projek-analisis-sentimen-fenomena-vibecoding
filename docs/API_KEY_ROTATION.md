# API Key Rotation & Monitoring

## File-file Utama

### 1. **models/GroqKeyManager.py**

Utility untuk manage Groq API keys

- `load_groq_env(env_path)` - Load environment variables
- `collect_groq_keys()` - Collect dan parse API keys dari `.envs`

### 2. **models/AutoLabelSentiment.py** ⭐ **MAIN**

Script untuk auto-labeling sentiment dengan API key rotation

```bash
python models/AutoLabelSentiment.py --input data/vibe_coding_dataset_ready.csv --output data/output.csv
```

**Fitur:**

- ✅ Automatic API key rotation saat rate limit/auth error
- ✅ Status monitoring: active dan failed keys
- ✅ Recovery mechanism: API key bisa di-reuse jika berhasil lagi
- ✅ Smart retry logic dengan max retries = jumlah API keys
- ✅ **Optimized rate limit handling** (2.5s delay = 24 req/min)

### 3. **models/testGroqKeys.py**

Test script untuk verify API keys configuration

```bash
python models/testGroqKeys.py
```

---

## GROQ Rate Limiting Explained

### Rate Limit Types

Groq memiliki **dua tipe limit yang berbeda**:

| Type                    | Free Tier       | Paid Tier     |
| ----------------------- | --------------- | ------------- |
| **Per Minute Limit**    | 30 requests/min | Higher        |
| **Monthly Quota**       | ~9000 req/day   | Based on plan |
| **Concurrent Requests** | 1 at a time     | More          |

### Penyebab Error 429 (Too Many Requests)

❌ **Jika sleep terlalu kecil:**

```
SLEEP_BETWEEN_CALLS = 0.5 detik
  → 1 / 0.5 = 2 requests/second
  → 2 * 60 = 120 requests/minute
  → GROQ limit = 30 requests/minute
  → 120 > 30 ❌ TERLALU CEPAT!

Result: Groq reject dengan 429 error
```

✅ **Dengan sleep yang tepat:**

```
SLEEP_BETWEEN_CALLS = 2.5 detik
  → 1 / 2.5 = 0.4 requests/second
  → 0.4 * 60 = 24 requests/minute
  → 24 < 30 ✓ SAFE!

Result: Smooth requests, no rate limit errors
```

### Mengapa Dashboard Masih Tampil Calls?

Saat error 429 terjadi:

1. Request **sudah dikirim** ke Groq (counted in dashboard)
2. Groq **reject** karena too fast
3. Client **kena error** dan trigger rotation
4. Dashboard hanya count **successful** requests

Contoh:

```
API[0]: 97 calls berhasil (sebelum hitting rate limit)
API[1]: 341 calls berhasil (sebelum hitting rate limit)
Sisa: Reject dengan 429 error → trigger rotation
```

---

## Optimal Settings

### Recommended SLEEP_BETWEEN_CALLS

| Value    | Req/sec | Req/min | Status      |
| -------- | ------- | ------- | ----------- |
| 0.5s     | 2.0     | 120     | ❌ Too fast |
| 1.0s     | 1.0     | 60      | ⚠️ At limit |
| 2.0s     | 0.5     | 30      | ⚠️ At limit |
| **2.5s** | **0.4** | **24**  | **✅ Safe** |
| 3.0s     | 0.33    | 20      | ✅ Safe     |

**Current setting:** `SLEEP_BETWEEN_CALLS = 2.5` ✓

---

## Cara Kerja API Key Rotation

```
[1] Start dengan API key index 0
    └─ Try labeling dengan 2.5s delay antar request...

[2] Jika error 429 (rate limit):
    ├─ Bukan karena terlalu cepat (sudah optimized)
    ├─ Mungkin API key sudah mencapai monthly quota
    ├─ Mark key sebagai FAILED
    ├─ Print status (active & failed keys)
    └─ Switch ke key lain yang masih available

[3] Jika semua keys FAILED:
    └─ Return "netral" sebagai fallback

[4] Jika labeling SUCCESS:
    ├─ Remove key dari FAILED (jika ada)
    └─ Continue ke text berikutnya
```

---

## Output Status Example

```
[INFO] Loaded 2 API keys untuk Groq

[STATUS] API Keys Summary:
  Current Active Key: index 0
  Available Keys: [0, 1]
  Failed Keys: []
  Total Available: 2/2
```

Ketika ada error:

```
[WARN] Error pada key index 0: Error code: 429 - rate limited
[WARN] API key index 0 kena rate limit, ditandai sebagai gagal

[STATUS] API Keys Summary:
  Current Active Key: index 1
  Available Keys: [1]
  Failed Keys: [0]
  Total Available: 1/2

[INFO] Mencoba dengan API key yang berbeda...
```

---

## Environment Variables Format

File `envs/.envs`:

```
API_KEY_GROQ="key1;key2;key3"
```

Supported formats:

- Semicolon-separated: `"key1;key2;key3"`
- Python list: `"[key1, key2, key3]"`
- Single key: `"key1"`

---

## Troubleshooting

### ❓ Masih kena rate limit error?

1. **Check SLEEP_BETWEEN_CALLS**

   ```python
   # Di models/AutoLabelSentiment.py, pastikan:
   SLEEP_BETWEEN_CALLS = 2.5  # atau lebih
   ```

2. **Check API key quota**

   - Login ke https://console.groq.com/keys
   - Lihat berapa requests sudah terpakai
   - Jika sudah habis bulan ini, tunggu bulan depan

3. **Use multiple API keys**

   - Pastikan punya 2+ API keys di envs/.envs
   - Script otomatis rotate ke key lain

4. **Reduce request frequency**
   ```python
   SLEEP_BETWEEN_CALLS = 3.0  # increase delay
   ```

### ❓ Script berhenti dengan "Semua API keys sudah mencapai rate limit"?

1. Semua API keys sudah mencapai monthly quota
2. Solution: Tunggu bulan depan atau request API key baru
