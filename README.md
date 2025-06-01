# Generator Judul Otomatis untuk Artikel Berita Berbahasa Indonesia Menggunakan Transformer

## Deskripsi Proyek
Banyak penulis artikel atau portal berita memerlukan waktu ekstra untuk membuat judul yang menarik dan relevan dengan isi artikel. Proyek ini bertujuan membangun model NLP yang dapat menghasilkan judul artikel secara otomatis dari isi artikel, sehingga mempercepat proses penulisan dan meningkatkan produktivitas.

## Dataset
Dataset yang digunakan berupa pasangan isi artikel dan judul dari berita online berbahasa Indonesia, seperti:
- **ID News Summarization Dataset**
- Hasil crawling dari situs berita (contoh: Kompas, Detik, CNN Indonesia)

## Pendekatan
Proyek ini menggunakan pendekatan NLP berbasis model transformer, khususnya dengan memanfaatkan model pra-latih **IndoT5** yang dirancang khusus untuk tugas-tugas pemrosesan bahasa alami dalam bahasa Indonesia. Model ini akan di-fine-tune menggunakan dataset yang berisi pasangan isi artikel dan judul, sehingga model dapat belajar menghasilkan judul yang sesuai dan relevan berdasarkan isi teks.

Proses fine-tuning dilakukan dengan skenario **text-to-text generation**, di mana input berupa isi artikel atau paragraf awal, dan output yang diharapkan adalah judul yang ringkas, menarik, serta mencerminkan inti dari artikel tersebut.

## Evaluasi
Evaluasi performa model dilakukan secara:
- **Kuantitatif**: menggunakan metrik **ROUGE** untuk mengukur kemiripan antara judul hasil model dan judul asli dari dataset.
- **Kualitatif**: melalui penilaian manual berdasarkan parameter kualitas judul seperti relevansi, kejelasan bahasa, dan daya tarik.

Dengan pendekatan ini, model tidak hanya diharapkan mampu meniru struktur judul yang baik, tetapi juga menghasilkan variasi judul baru yang kreatif dan layak digunakan dalam praktik jurnalistik.

## Output yang Diharapkan
Model yang dapat menghasilkan judul artikel secara otomatis dari isi artikel, dengan kualitas judul berdasarkan parameter terukur untuk mendukung validitas hasil. 