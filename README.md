# Perbandingan Deteksi Tepi: Sobel Biasa vs Adaptive Sobel

Aplikasi Streamlit interaktif untuk membandingkan dua metode deteksi tepi: Sobel biasa dan Adaptive Sobel.

## Deskripsi

Aplikasi ini memungkinkan pengguna untuk mengunggah gambar dan membandingkan hasil deteksi tepi menggunakan dua metode berbeda:
1. **Sobel Biasa**: Filter deteksi tepi standar yang menggunakan threshold global
2. **Adaptive Sobel**: Modifikasi dari Sobel yang menggunakan threshold adaptif berdasarkan karakteristik lokal gambar

Perbandingan ini membantu pengguna memahami kelebihan dan kekurangan masing-masing metode dalam berbagai kondisi gambar.

## Fitur

- Upload gambar dari perangkat pengguna
- Tampilan gambar asli
- Pemrosesan gambar dengan dua metode deteksi tepi
- Tampilan hasil deteksi tepi secara berdampingan (side-by-side)
- Histogram dari masing-masing hasil untuk analisis visual
- Metrik evaluasi (Edge Pixel Count, PSNR, SSIM)
- Distribusi Intensitas Tepi untuk analisis statistik
- Grafik Profil Kontras untuk analisis sensitivitas kontras lokal
- Overlay Tepi pada gambar asli untuk evaluasi visual
- Penjelasan singkat perbedaan metode secara teori
- Antarmuka yang bersih, sederhana, dan interaktif

## Metode yang Digunakan

### Sobel Biasa
Filter Sobel standar adalah operator deteksi tepi yang menggunakan konvolusi dengan kernel tertentu untuk menghitung gradien intensitas gambar. Filter ini mendeteksi perubahan intensitas pada arah horizontal dan vertikal, kemudian menggabungkannya untuk mendapatkan magnitude gradien. Filter ini menggunakan threshold global yang sama untuk seluruh gambar.

### Adaptive Sobel
Adaptive Sobel adalah modifikasi dari filter Sobel standar yang menggunakan thresholding adaptif. Metode ini menyesuaikan nilai ambang batas (threshold) berdasarkan karakteristik lokal dari gambar, sehingga dapat beradaptasi dengan variasi intensitas di berbagai area gambar. Hal ini membuat deteksi tepi lebih efektif pada gambar dengan pencahayaan yang tidak merata atau kontras yang bervariasi.

## Cara Menggunakan

1. Pastikan Python dan library yang diperlukan sudah terinstal
2. Clone repositori ini
3. Instal dependensi dengan menjalankan:
   ```
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi dengan perintah:
   ```
   streamlit run app.py
   ```
5. Buka browser dan akses URL yang ditampilkan (biasanya http://localhost:8501)
6. Upload gambar dan lihat perbandingan hasil deteksi tepi

## Metrik Evaluasi

Aplikasi ini menghitung beberapa metrik untuk membandingkan hasil deteksi tepi:

- **Edge Pixel Count**: Jumlah piksel yang terdeteksi sebagai tepi
- **PSNR (Peak Signal-to-Noise Ratio)**: Mengukur kualitas gambar hasil dibandingkan dengan gambar asli
- **SSIM (Structural Similarity Index)**: Mengukur kesamaan struktural antara gambar hasil dan gambar asli

## Persyaratan

- Python 3.7+
- Streamlit
- OpenCV (cv2)
- NumPy
- Matplotlib
- scikit-image

Anda dapat menginstal semua dependensi dengan menjalankan:
```
pip install streamlit opencv-python numpy matplotlib scikit-image
```

Atau menggunakan file requirements.txt yang disediakan:
```
pip install -r requirements.txt
```

## File Requirements.txt

Untuk memudahkan instalasi, berikut adalah isi file requirements.txt:

```
streamlit>=1.12.0
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-image>=0.18.0
pillow>=8.0.0
```

## Kontribusi

Kontribusi untuk meningkatkan aplikasi ini sangat diterima. Silakan buat pull request atau buka issue untuk diskusi.

## Cara Menjalankan Aplikasi

Untuk menjalankan aplikasi ini, Anda dapat menggunakan perintah berikut di terminal:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan berjalan di browser Anda dan Anda dapat mulai mengunggah gambar untuk membandingkan metode deteksi tepi Sobel biasa dan Adaptive Sobel.

Semoga aplikasi ini bermanfaat untuk kebutuhan Anda dalam membandingkan kedua metode deteksi tepi tersebut!

        