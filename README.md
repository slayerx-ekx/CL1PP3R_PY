# CL1PP3R_PY



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

## 🧠 3. Analisis Subtitle

### 💡 Contoh Prompt untuk AI (ChatGPT / LLM):

Analyze this subtitle file and generate 3-5 short-form video clips optimized for TikTok/YouTube Shorts with:
1. Viral potential scoring (1–10)
2. Timestamps (MM:SS - MM:SS)
3. Short, catchy clip titles
4. Duration max 60 seconds per clip
Return result in JSON format:
{
  "video_title": "Apa yang Gua Harap Gua Tau di Umur 17",
  "duration": "16:27",
  "theme": "Life Advice / Entrepreneurship / Personal Growth",
  "clips": [
    {
      "clip_title": "Uang Bukan Segalanya, Tapi Alat!",
      "timestamp": "02:53 - 03:30",
      "viral_score": 8.8
    },
    {
      "clip_title": "Jangan Jadi Superman, Bangun Super Team!",
      "timestamp": "05:07 - 05:40",
      "viral_score": 9.0
    }
  ]
}


### 📦 Contoh Output JSON:

```json
{
  "video_title": "Apa yang Gua Harap Gua Tau di Umur 17",
  "duration": "16:27",
  "theme": "Life Advice / Entrepreneurship / Personal Growth",
  "clips": [
    {
      "clip_title": "Uang Bukan Segalanya, Tapi Alat!",
      "timestamp": "02:53 - 03:30",
      "viral_score": 8.8
    },
    {
      "clip_title": "Jangan Jadi Superman, Bangun Super Team!",
      "timestamp": "05:07 - 05:40",
      "viral_score": 9.0
    }
  ]
}
```

---

## 🐍 4. Penggunaan JSON di Aplikasi Python


### 📂 Memilih File JSON Lewat Browser File (GUI)

### ⚙️ Proses JSON di Aplikasi Python

Gunakan data JSON untuk:


---
