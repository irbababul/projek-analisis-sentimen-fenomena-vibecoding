# Nama Label & Mapping
- `positive` => label_id = 0
- `negative` => label_id = 1
- `neutral` => label_id = 2

# Fokus Sentimen
Yang dinilai adalah **sikap terhadap fenomena vibecoding** 
(Bukan hanya sekadar terhadap video/pembicara, kecuali memang komentarnya tentang itu)

# Aturan
## 1. Positive (label_id = 2)
Komentar yang menunjukkan sikap positif / mendukung / antusias terhadap fenomena vibe coding atau dampaknya.

**ciri-ciri**
- Merasa terbantu, termotivasi dan lebih produktif
- Melihat vibe coding sebagai hal yang keren, positif, menarik dan membuka peluang

**kata/frasa contoh**:
- keren
- ngebantu banget
- jadi semangat
- bagus
- worth it
- memudahkan
- makin gampang belajar

**contoh singkat**:
- gokil sih, bikin gue berani nyoba AI buat ngoding
- vibe coding gini cocok banget buat orang visual

## 2. Negative (label_id = 1)
Komentar yang menunjukkan sikap negatif / khawatir / menolak terhadap vibe coding atau dampaknya.

**ciri-ciri**:
- Menyebut vibe coding berbahaya, menyesatkan, bikin malas belajar dasar
- Takut kehilangan pekerjaan / turunnya kualitas engineer
- Menilai fenomena ini sebagai sesuatu yang buruk / tidak layak

**contoh singkat**:
- ini bahaya banget buat pemula, jadi gak belajar dasar
- vibe coding gini bikin standar programmer jadi turun

## 3. Netral (label_id = 2)
Komentar yang:
- **deskriptif** / **observasi** / **bertanya** tanpa stance yang jelas
- infromatif, sekadar sharing, atau campuran tapi **tidak jelas dominan ke positif atau negatif**
- komentar yang tidak relevan langsung ke fenomena vibe coding

**contoh**:
- bedanya vibe coding sama pakai copilot biasa apa?
- di kantor gue belum ada pakai beginian, masih pakai tradisional aja
- komentar bercanda random yang tidak menilai vibe coding

# Edge Cases (Penting)
- Campuran (ada plus dan minus):
-- Kalau jelas lebih condong ke salah satu => Ambil yang dominan
-- Kalau benar-benar seimbang => `netral`

- Sarkas:
-- kalau konteksknya mengejek vibe coding => `negatif`

- Komentar yang hanya *"mantap bang"*, tanpa konteks vibe coding => `netral`