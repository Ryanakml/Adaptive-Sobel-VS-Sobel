import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io

# Fungsi untuk menerapkan Sobel biasa
def apply_regular_sobel(image, ksize=3):
    # Konversi ke grayscale jika gambar berwarna
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Terapkan filter Sobel pada sumbu x dan y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Hitung magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalisasi ke rentang 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return magnitude

# Fungsi untuk menerapkan Adaptive Sobel
def apply_adaptive_sobel(image, block_size=11, c=2):
    # Konversi ke grayscale jika gambar berwarna
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Terapkan filter Sobel pada sumbu x dan y
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Hitung magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalisasi ke rentang 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Terapkan thresholding adaptif pada magnitude
    adaptive_threshold = cv2.adaptiveThreshold(
        magnitude,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c
    )
    
    return adaptive_threshold

# Fungsi untuk menghitung histogram
def calculate_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist

# Fungsi untuk menghitung metrik evaluasi
def calculate_metrics(original, processed):
    # Hitung jumlah piksel tepi
    edge_pixel_count = np.sum(processed > 0)
    
    # Hitung PSNR jika memungkinkan (hindari pembagian dengan nol)
    try:
        psnr_value = psnr(original, processed)
    except:
        psnr_value = "Tidak dapat dihitung"
    
    # Hitung SSIM jika memungkinkan
    try:
        ssim_value = ssim(original, processed)
    except:
        ssim_value = "Tidak dapat dihitung"
    
    return {
        "Edge Pixel Count": edge_pixel_count,
        "PSNR": psnr_value,
        "SSIM": ssim_value
    }

# Fungsi untuk menampilkan histogram
def plot_histogram(hist, title):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hist)
    ax.set_xlim([0, 256])
    ax.set_title(title)
    ax.set_xlabel('Intensitas Piksel')
    ax.set_ylabel('Jumlah Piksel')
    return fig

# Judul aplikasi
st.title("Perbandingan Deteksi Tepi: Sobel Biasa vs Adaptive Sobel")

# Penjelasan singkat
st.markdown("""
## Penjelasan Metode

### Sobel Biasa
Filter Sobel standar adalah operator deteksi tepi yang menggunakan konvolusi dengan kernel tertentu untuk menghitung gradien intensitas gambar. 
Filter ini mendeteksi perubahan intensitas pada arah horizontal dan vertikal, kemudian menggabungkannya untuk mendapatkan magnitude gradien.
Filter ini menggunakan threshold global yang sama untuk seluruh gambar.

### Adaptive Sobel
Adaptive Sobel adalah modifikasi dari filter Sobel standar yang menggunakan thresholding adaptif. 
Metode ini menyesuaikan nilai ambang batas (threshold) berdasarkan karakteristik lokal dari gambar, 
sehingga dapat beradaptasi dengan variasi intensitas di berbagai area gambar. 
Hal ini membuat deteksi tepi lebih efektif pada gambar dengan pencahayaan yang tidak merata atau kontras yang bervariasi.
""")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar untuk diproses", type=["jpg", "jpeg", "png"])

# Parameter untuk Adaptive Sobel
st.sidebar.header("Parameter")
block_size = st.sidebar.slider("Block Size (Adaptive Sobel)", 3, 99, 11, step=2)  # Harus ganjil
c_value = st.sidebar.slider("C Value (Adaptive Sobel)", 0, 10, 2)
sobel_ksize = st.sidebar.selectbox("Kernel Size (Sobel)", [3, 5, 7])

if uploaded_file is not None:
    # Baca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi BGR ke RGB untuk tampilan
    
    # Tampilkan gambar asli
    st.header("Gambar Asli")
    st.image(image_rgb, caption="Gambar Input", use_container_width=True)
    
    # Proses gambar dengan kedua metode
    regular_sobel = apply_regular_sobel(image, ksize=sobel_ksize)
    adaptive_sobel = apply_adaptive_sobel(image, block_size=block_size, c=c_value)
    
    # Tampilkan hasil deteksi tepi secara berdampingan
    st.header("Hasil Deteksi Tepi")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sobel Biasa")
        st.image(regular_sobel, caption="Hasil Sobel Biasa", use_container_width=True)
    
    with col2:
        st.subheader("Adaptive Sobel")
        st.image(adaptive_sobel, caption="Hasil Adaptive Sobel", use_container_width=True)
    
    # Hitung dan tampilkan histogram
    st.header("Histogram")
    hist_regular = calculate_histogram(regular_sobel)
    hist_adaptive = calculate_histogram(adaptive_sobel)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.pyplot(plot_histogram(hist_regular, "Histogram Sobel Biasa"))
    
    with col2:
        st.pyplot(plot_histogram(hist_adaptive, "Histogram Adaptive Sobel"))
    
    # Hitung dan tampilkan metrik evaluasi
    st.header("Metrik Evaluasi")
    
    # Konversi gambar asli ke grayscale untuk perbandingan
    if len(image.shape) == 3:
        gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_original = image
    
    metrics_regular = calculate_metrics(gray_original, regular_sobel)
    metrics_adaptive = calculate_metrics(gray_original, adaptive_sobel)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metrik Sobel Biasa")
        for metric, value in metrics_regular.items():
            st.write(f"{metric}: {value}")
    
    with col2:
        st.subheader("Metrik Adaptive Sobel")
        for metric, value in metrics_adaptive.items():
            st.write(f"{metric}: {value}")
    
    # Perbandingan langsung
    st.header("Perbandingan Langsung")
    
    # Hitung perbedaan jumlah piksel tepi
    edge_diff = metrics_adaptive["Edge Pixel Count"] - metrics_regular["Edge Pixel Count"]
    st.write(f"Perbedaan Jumlah Piksel Tepi: {edge_diff} piksel")
    
    # Tampilkan kesimpulan berdasarkan metrik
    st.subheader("Kesimpulan")
    
    if metrics_adaptive["Edge Pixel Count"] > metrics_regular["Edge Pixel Count"]:
        st.write("Adaptive Sobel mendeteksi lebih banyak tepi dibandingkan Sobel biasa.")
    else:
        st.write("Sobel biasa mendeteksi lebih banyak tepi dibandingkan Adaptive Sobel.")
    
    st.write("""
    **Catatan Interpretasi:**
    - **Edge Pixel Count**: Jumlah piksel yang terdeteksi sebagai tepi. Nilai lebih tinggi menunjukkan lebih banyak tepi terdeteksi.
    - **PSNR**: Peak Signal-to-Noise Ratio. Nilai lebih tinggi menunjukkan kualitas gambar yang lebih baik dibandingkan dengan gambar asli.
    - **SSIM**: Structural Similarity Index. Nilai lebih tinggi (mendekati 1) menunjukkan kesamaan struktural yang lebih baik dengan gambar asli.
    """)
else:
    st.info("Silakan upload gambar untuk memulai perbandingan.")