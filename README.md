# CL1PP3R_PY
---

# 🎬 CLIP AUTOMATION — Smart Video Clipper

**CLIP AUTOMATION** adalah aplikasi berbasis GUI (Python) yang dirancang untuk memotong video panjang menjadi klip-klip pendek secara otomatis dan efisien. Cocok untuk content creator, editor profesional, maupun tim media sosial.

---

## 🚀 Fitur Utama

### 🔪 Pemotongan Otomatis

* Potong video berdasarkan durasi tetap
* Import konfigurasi klip dari file `.json`

### 🎥 Sumber Video Fleksibel

* Mendukung berbagai format lokal: `.mp4`, `.mov`, `.avi`, `.mkv`
* Download langsung dari YouTube (dengan URL)

### 📱 Optimasi untuk Shorts

* Konversi otomatis ke format vertikal (9:16)
* Subtitle otomatis dengan 16+ gaya (Karaoke, Bold, Highlighted, dll.)
* Pilihan kualitas model Whisper (`tiny` hingga `large`)

### 🗂️ Manajemen Proyek

* Log proses real-time
* Progress bar visual
* Penamaan file otomatis berdasarkan klip

---

## 🛠️ Teknologi yang Digunakan

* **PyQt5** – Antarmuka GUI yang modern dan responsif
* **MoviePy** – Pemrosesan dan pemotongan video
* **Whisper (OpenAI)** – Transkripsi audio ke teks
* **Hugging Face Transformers** – Analisis sentimen dan penilaian konten
* **FFmpeg** – Rendering video lanjutan dan konversi format

---

## 💻 Cara Instalasi

1. Pastikan Python 3.8 atau lebih tinggi telah terpasang
2. Install FFmpeg dan tambahkan ke PATH
3. Clone repository ini dan jalankan:

   ```
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:

   ```
   python CLIP_AUTOMATION.py
   ```

---

## 🌟 Keunggulan

* ✅ 100% otomatis — Tanpa perlu edit manual
* 🔓 Open-source — Mudah dimodifikasi sesuai kebutuhan

---

## 🎯 Cocok Untuk

* YouTube Shorts & TikTok Creator
* Tim konten media sosial
* Pembuat konten edukasi dan training

---


# 🎬 Panduan Lengkap: Membuat Klip Pendek dari Video YouTube

Panduan ini menjelaskan **langkah demi langkah** bagaimana:

1. Memilih video YouTube yang cocok
2. Mengunduh subtitle
3. Menganalisis subtitle untuk klip pendek viral
4. Menyimpan hasil dalam format JSON
5. Menggunakannya dalam aplikasi Python

---

## 📌 Format Output JSON yang Diinginkan

```json
{
  "video_title": "Judul Video",
  "duration": "HH:MM",
  "theme": "Kategori/Konten",
  "clips": [
    {
      "clip_title": "Judul Klip",
      "timestamp": "MM:SS - MM:SS",
      "viral_score": X.X
    }
  ]
}
```

---

## 🔍 1. Pencarian Video YouTube

### 🎯 Kriteria Video Ideal untuk Klip Pendek:

* Durasi antara **5–20 menit**
* Tema edukatif, motivasi, storytelling, atau inspiratif
* Memiliki banyak **momen powerful atau emosional**
* Speaker berbicara jelas dan tidak terlalu cepat

### ✏️ Cara Menyalin Link Video:

1. Buka YouTube dan cari video yang sesuai
2. Klik tombol **Bagikan** (Share)
3. Salin tautan video

---

## 📝 2. Pengunduhan Subtitle

1. Buka bowser dan cari website untuk melakukan download subtitle youtube

### ✅ Rekomendasi Website/Tools:

* [DownloadYouTubeSubtitles.com](https://downsub.com)
* [DownSub](https://downsub.com)
* [yt-dlp](https://github.com/yt-dlp/yt-dlp) (command-line tool)

### 🔧 Langkah-langkah Download Subtitle (Format .srt/.txt):

1. Tempel link video ke situs [downsub.com](https://downsub.com)
2. Pilih subtitle **auto-generated** (atau manual jika tersedia)
3. Download file dalam format **.srt** atau **.txt**
4. Simpan file di direktori proyek

---
### 💡 Lakukan Prompt untuk AI (ChatGPT / LLM) Untuk melakukan generate .json file:

Analyze this subtitle file and generate 3-5 short-form video clips optimized for TikTok/YouTube Shorts with:
1. Viral potential scoring (1–10)
2. Timestamps (MM:SS - MM:SS)
3. Short, catchy clip titles
4. Duration max 60 seconds per clip
Return result in JSON format:
```
{
  "video_title": "Judul Video",
  "duration": "HH:MM",
  "theme": "Kategori/Konten",
  "clips": [
    {
      "clip_title": "Judul Klip",
      "timestamp": "MM:SS - MM:SS",
      "viral_score": X.X
    }
  ]
}
```

### 📂 Memilih File JSON Lewat Browser File (GUI)

### ⚙️ Proses JSON di Aplikasi Python

Gunakan data JSON untuk:
1. Pemotongan video, sesuai dengan json file
2. Menambahkan judul di clip yang sudah sesuai sama dengan json file
