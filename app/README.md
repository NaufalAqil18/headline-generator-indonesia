# Article Title Generator AI

Aplikasi web untuk mengenerate judul artikel secara otomatis menggunakan tiga model AI:
- **Gemini (Google LLM API)**
- **BERT2BERT Indonesian Summarization (cahya/bert2bert-indonesian-summarization)**
- **Custom Tuning Model (local, fine-tuned)**

Tampilan modern, dark mode, dan hasil perbandingan model yang mudah dibaca.

---

## ✨ Fitur Utama
- Input artikel dalam Bahasa Indonesia
- Generate judul dari 3 model AI sekaligus
- Perbandingan hasil judul secara visual (vertikal, card)
- UI profesional, responsif, dan mudah digunakan
- Mendukung GPU/CPU (PyTorch)

---

## 🛠️ Tech Stack
- **Frontend:** Streamlit (Python)
- **Backend:** FastAPI (Python)
- **Model:**
  - Google Gemini API
  - HuggingFace Transformers (BERT2BERT)
  - Custom model (local, folder `models/best_model_epoch_2`)

---

## 🚀 Instalasi & Setup
1. **Clone repo ini:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Install dependensi:**
   ```bash
   pip install -r gemini/requirements.txt
   pip install -r gemini/backend/requirements.txt
   ```
   > Pastikan juga install `torch` yang sesuai dengan device kamu.

3. **Siapkan file .env untuk Gemini API:**
   - Buat file `.env` di `gemini/backend/` dengan isi:
     ```
     GEMINI_API_KEY=ISI_API_KEY_MU
     ```

4. **Pastikan folder model lokal sudah ada:**
   - `gemini/models/best_model_epoch_2/` (isi: model, config, tokenizer, dsb)

---

## 🏃 Cara Menjalankan
### 1. Jalankan Backend (FastAPI)
Masuk ke folder /backend lalu jalankan perintah ini
```bash
uvicorn main:app --reload
```
- Backend berjalan di `http://localhost:8000`

### 2. Jalankan Frontend (Streamlit)
Masuk ke folder gemini
```bash
streamlit run app.py
```
- Frontend akan terbuka di browser (`http://localhost:8501`)

---

## 💡 Contoh Penggunaan
1. Masukkan artikel pada kolom input.
2. Klik **Generate Titles**.
3. Lihat dan bandingkan hasil judul dari Gemini, BERT, dan Tuning Model.

---

## 📁 Struktur Folder
```
├── gemini/
│   ├── app.py                # Frontend Streamlit
│   ├── backend/
│   │   └── main.py           # Backend FastAPI
│   │   └── .env              # (isi API key Gemini)
│   ├── models/
│   │   └── best_model_epoch_2/ # Model custom (local)
│   └── requirements.txt      # Dependensi utama
├── README.md
```

---

## 👥 Kontributor
- Naufal Aqil
- Shofia Nurul Huda

---

## 📄 Lisensi
MIT (atau sesuai kebutuhan) 