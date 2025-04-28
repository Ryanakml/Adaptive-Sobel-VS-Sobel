import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import io
import os
import scipy.io as sio  # For reading .mat files

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

# New function to calculate edge intensity distribution
def calculate_edge_intensity_distribution(image):
    """Calculate the distribution of edge intensities"""
    # Only consider pixels that are edges (non-zero)
    edge_pixels = image[image > 0]
    
    if len(edge_pixels) == 0:
        return None, 0, 0, 0
    
    # Calculate statistics
    mean_intensity = np.mean(edge_pixels)
    std_intensity = np.std(edge_pixels)
    median_intensity = np.median(edge_pixels)
    
    # Create histogram of edge intensities
    hist, bins = np.histogram(edge_pixels, bins=50, range=(0, 255))
    
    return hist, mean_intensity, std_intensity, median_intensity

# New function to calculate local contrast around edges
# Fungsi yang diperbarui untuk menghitung kontras lokal
def calculate_local_contrast(image, edge_map, window_size=5):
    """Calculate the local contrast around detected edges"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Create a mask of edge pixels
    edge_mask = edge_map > 0
    
    # If no edges detected, return None
    if not np.any(edge_mask):
        return None, 0, None, None
    
    # Initialize array to store contrast values
    contrast_values = []
    
    # Get coordinates of edge pixels
    edge_coords = np.where(edge_mask)
    
    # Sample up to 1000 edge pixels for efficiency
    sample_size = min(1000, len(edge_coords[0]))
    indices = np.random.choice(len(edge_coords[0]), sample_size, replace=False)
    
    half_window = window_size // 2
    
    # For each sampled edge pixel, calculate local contrast
    for idx in indices:
        y, x = edge_coords[0][idx], edge_coords[1][idx]
        
        # Define window boundaries with padding for image edges
        y_min = max(0, y - half_window)
        y_max = min(gray.shape[0], y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(gray.shape[1], x + half_window + 1)
        
        # Extract window
        window = gray[y_min:y_max, x_min:x_max]
        
        # Calculate local contrast (max - min) in window
        if window.size > 0:
            local_contrast = np.max(window) - np.min(window)
            contrast_values.append(local_contrast)
    
    if not contrast_values:
        return None, 0, None, None
    
    # Calculate average local contrast
    avg_contrast = np.mean(contrast_values)
    
    # Create histogram of contrast values
    hist, bins = np.histogram(contrast_values, bins=50, range=(0, 255))
    
    # Pilih beberapa garis profil untuk analisis kontras
    # Pilih 3 baris acak yang memiliki tepi
    profile_rows = []
    profile_data = []
    
    # Cari baris yang memiliki tepi
    rows_with_edges = np.unique(edge_coords[0])
    if len(rows_with_edges) > 0:
        # Pilih hingga 3 baris acak
        selected_rows = np.random.choice(rows_with_edges, min(3, len(rows_with_edges)), replace=False)
        
        for row in selected_rows:
            # Ambil data intensitas sepanjang baris
            row_data = gray[row, :]
            # Hitung gradien (perubahan intensitas) sepanjang baris
            gradient = np.abs(np.diff(row_data.astype(np.float32)))
            # Tambahkan padding agar ukurannya sama dengan row_data
            gradient = np.append(gradient, 0)
            
            profile_rows.append(row)
            profile_data.append((row_data, gradient))
    
    return hist, avg_contrast, profile_rows, profile_data

# New function to create edge map overlay
def create_edge_overlay(image, edge_map, color=(0, 255, 0)):
    """Create an overlay of edges on the original image"""
    if len(image.shape) == 3:
        overlay = image.copy()
    else:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Create a mask where edges are detected
    mask = edge_map > 0
    
    # Apply the color to edge pixels
    overlay[mask] = color
    
    return overlay

# Function to load ground truth from .mat file
def load_ground_truth(image_name):
    """
    Load ground truth edge map from .mat file
    """
    # Check in test, train, and val directories
    for subset in ['test', 'train', 'val']:
        gt_path = os.path.join('./BIPED/imgs/train', 
                              subset, f"{image_name}.mat")
        if os.path.exists(gt_path):
            try:
                # Load .mat file
                mat_contents = sio.loadmat(gt_path)
                
                # BSD dataset typically stores ground truth in 'groundTruth' field
                # Each image may have multiple ground truth annotations
                gt_data = mat_contents['groundTruth']
                
                # Get the first ground truth (you could also average multiple annotations)
                # The 'Boundaries' field contains the edge map
                edge_map = gt_data[0, 0]['Boundaries'][0, 0]
                
                # Convert to binary image (0 or 255)
                edge_map = (edge_map > 0).astype(np.uint8) * 255
                
                return edge_map
            except Exception as e:
                st.error(f"Error loading ground truth: {e}")
                return None
    
    return None

# Modified function to calculate metrics with ground truth
def calculate_metrics(original, processed, ground_truth=None):
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
    
    # If ground truth is provided, use it for evaluation
    if ground_truth is not None:
        # Resize ground truth to match processed image if needed
        if ground_truth.shape != processed.shape:
            ground_truth = cv2.resize(ground_truth, (processed.shape[1], processed.shape[0]))
        
        # Binarize ground truth and processed image
        gt_binary = ground_truth > 0
        proc_binary = processed > 0
        
        # Calculate True Positive, False Positive, False Negative
        true_positive = np.sum(proc_binary & gt_binary)
        false_positive = np.sum(proc_binary & ~gt_binary)
        false_negative = np.sum(~proc_binary & gt_binary)
        
    else:
        # Use synthetic ground truth as before
        sobelx = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, synthetic_gt = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        
        # Calculate metrics using synthetic ground truth
        gt_binary = synthetic_gt > 0
        proc_binary = processed > 0
        
        true_positive = np.sum(proc_binary & gt_binary)
        false_positive = np.sum(proc_binary & ~gt_binary)
        false_negative = np.sum(~proc_binary & gt_binary)
    
    # Calculate Precision, Recall, and F1-Score
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "Edge Pixel Count": edge_pixel_count,
        "PSNR": psnr_value,
        "SSIM": ssim_value,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score
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

# Add option to choose between upload or sample images
image_source = st.radio(
    "Pilih Sumber Gambar:",
    ("Upload Gambar", "Gunakan Gambar Sampel")
)

# Path to sample images
sample_images_dir = '/Users/ryanakmalpasya/Documents/[1] BS/[2] Freelance/[3] PROJECTS/Adaptive Sobel vs Common Sobel/sample_images'

# Variable to store the image
image = None
image_rgb = None
file_name = None

if image_source == "Upload Gambar":
    # Upload gambar
    uploaded_file = st.file_uploader("Pilih gambar untuk diproses", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Baca gambar
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi BGR ke RGB untuk tampilan
        file_name = os.path.splitext(uploaded_file.name)[0]
else:
    # List sample images
    if os.path.exists(sample_images_dir) and os.path.isdir(sample_images_dir):
        sample_files = [f for f in os.listdir(sample_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if sample_files:
            selected_sample = st.selectbox("Pilih Gambar Sampel:", sample_files)
            
            if selected_sample:
                sample_path = os.path.join(sample_images_dir, selected_sample)
                image = cv2.imread(sample_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                file_name = os.path.splitext(selected_sample)[0]
                
                st.success(f"Gambar sampel dipilih: {selected_sample}")
        else:
            st.error("Tidak ada gambar sampel ditemukan. Silakan tambahkan gambar ke folder sample_images.")
    else:
        st.error(f"Folder sample_images tidak ditemukan di {sample_images_dir}. Silakan buat folder dan tambahkan gambar sampel.")

# Parameter untuk Adaptive Sobel
st.sidebar.header("Parameter")
block_size = st.sidebar.slider("Block Size (Adaptive Sobel)", 3, 99, 11, step=2)  # Harus ganjil
c_value = st.sidebar.slider("C Value (Adaptive Sobel)", 0, 10, 2)
sobel_ksize = st.sidebar.selectbox("Kernel Size (Sobel)", [3, 5, 7])

# Remove ground truth option since we're using alternative metrics
# use_ground_truth = st.sidebar.checkbox("Use Ground Truth (if available)", False)

# Process the image if it exists
if image is not None:
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
    
    # Konversi gambar asli ke grayscale untuk perbandingan
    if len(image.shape) == 3:
        gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_original = image
    
    # NEW SECTION: Edge Intensity Distribution
    st.header("Distribusi Intensitas Tepi")
    
    # Calculate edge intensity distributions
    reg_hist, reg_mean, reg_std, reg_median = calculate_edge_intensity_distribution(regular_sobel)
    adap_hist, adap_mean, adap_std, adap_median = calculate_edge_intensity_distribution(adaptive_sobel)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sobel Biasa")
        st.write(f"Rata-rata Intensitas: {reg_mean:.2f}")
        st.write(f"Standar Deviasi: {reg_std:.2f}")
        st.write(f"Median Intensitas: {reg_median:.2f}")
        
        if reg_hist is not None:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(range(len(reg_hist)), reg_hist)
            ax.set_title("Distribusi Intensitas Tepi - Sobel Biasa")
            ax.set_xlabel("Intensitas")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)
    
    with col2:
        st.subheader("Adaptive Sobel")
        st.write(f"Rata-rata Intensitas: {adap_mean:.2f}")
        st.write(f"Standar Deviasi: {adap_std:.2f}")
        st.write(f"Median Intensitas: {adap_median:.2f}")
        
        if adap_hist is not None:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(range(len(adap_hist)), adap_hist)
            ax.set_title("Distribusi Intensitas Tepi - Adaptive Sobel")
            ax.set_xlabel("Intensitas")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)
    
    # Di bagian aplikasi Streamlit, setelah bagian "Sensitivitas Kontras Lokal"
    # NEW SECTION: Local Contrast Sensitivity
    st.header("Sensitivitas Kontras Lokal")
    
    # Calculate local contrast
    reg_contrast_hist, reg_avg_contrast, reg_profile_rows, reg_profile_data = calculate_local_contrast(image, regular_sobel)
    adap_contrast_hist, adap_avg_contrast, adap_profile_rows, adap_profile_data = calculate_local_contrast(image, adaptive_sobel)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sobel Biasa")
        st.write(f"Rata-rata Kontras Lokal: {reg_avg_contrast:.2f}")
        
        if reg_contrast_hist is not None:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(range(len(reg_contrast_hist)), reg_contrast_hist)
            ax.set_title("Distribusi Kontras Lokal - Sobel Biasa")
            ax.set_xlabel("Nilai Kontras")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)
    
    with col2:
        st.subheader("Adaptive Sobel")
        st.write(f"Rata-rata Kontras Lokal: {adap_avg_contrast:.2f}")
        
        if adap_contrast_hist is not None:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(range(len(adap_contrast_hist)), adap_contrast_hist)
            ax.set_title("Distribusi Kontras Lokal - Adaptive Sobel")
            ax.set_xlabel("Nilai Kontras")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)
    
    # NEW SECTION: Contrast Profile Visualization
    st.header("Grafik Profil Kontras")
    st.write("""
    Grafik di bawah ini menunjukkan profil kontras (perubahan gradien) di sepanjang beberapa baris gambar yang berisi tepi.
    Puncak yang lebih tinggi menunjukkan perubahan intensitas yang lebih tajam, yang biasanya merupakan tepi yang kuat.
    """)
    
    # Visualisasi profil kontras untuk Sobel Biasa
    if reg_profile_data and len(reg_profile_data) > 0:
        st.subheader("Profil Kontras - Sobel Biasa")
        fig, axs = plt.subplots(len(reg_profile_data), 1, figsize=(10, 3*len(reg_profile_data)))
        
        # Jika hanya ada satu baris, axs tidak akan menjadi array
        if len(reg_profile_data) == 1:
            axs = [axs]
        
        for i, (row, (intensity, gradient)) in enumerate(zip(reg_profile_rows, reg_profile_data)):
            # Plot intensitas dan gradien
            axs[i].plot(intensity, 'b-', alpha=0.5, label='Intensitas')
            axs[i].plot(gradient, 'r-', label='Gradien (Kontras Lokal)')
            
            # Tandai piksel tepi pada baris ini
            edge_positions = np.where(regular_sobel[row, :] > 0)[0]
            if len(edge_positions) > 0:
                axs[i].plot(edge_positions, intensity[edge_positions], 'go', label='Tepi Terdeteksi')
            
            axs[i].set_title(f'Baris {row}')
            axs[i].set_xlabel('Posisi Piksel')
            axs[i].set_ylabel('Nilai')
            axs[i].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("Tidak dapat membuat profil kontras untuk Sobel Biasa.")
    
    # Visualisasi profil kontras untuk Adaptive Sobel
    if adap_profile_data and len(adap_profile_data) > 0:
        st.subheader("Profil Kontras - Adaptive Sobel")
        fig, axs = plt.subplots(len(adap_profile_data), 1, figsize=(10, 3*len(adap_profile_data)))
        
        # Jika hanya ada satu baris, axs tidak akan menjadi array
        if len(adap_profile_data) == 1:
            axs = [axs]
        
        for i, (row, (intensity, gradient)) in enumerate(zip(adap_profile_rows, adap_profile_data)):
            # Plot intensitas dan gradien
            axs[i].plot(intensity, 'b-', alpha=0.5, label='Intensitas')
            axs[i].plot(gradient, 'r-', label='Gradien (Kontras Lokal)')
            
            # Tandai piksel tepi pada baris ini
            edge_positions = np.where(adaptive_sobel[row, :] > 0)[0]
            if len(edge_positions) > 0:
                axs[i].plot(edge_positions, intensity[edge_positions], 'go', label='Tepi Terdeteksi')
            
            axs[i].set_title(f'Baris {row}')
            axs[i].set_xlabel('Posisi Piksel')
            axs[i].set_ylabel('Nilai')
            axs[i].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.write("Tidak dapat membuat profil kontras untuk Adaptive Sobel.")
    
    # Perbandingan langsung profil kontras
    if reg_profile_data and adap_profile_data and len(reg_profile_data) > 0 and len(adap_profile_data) > 0:
        st.subheader("Perbandingan Profil Kontras")
        st.write("""
        Grafik di bawah ini membandingkan profil kontras antara Sobel Biasa dan Adaptive Sobel pada baris yang sama.
        Perhatikan bagaimana kedua metode merespons perubahan kontras yang berbeda.
        """)
        
        # Pilih baris pertama dari masing-masing untuk perbandingan
        reg_row = reg_profile_rows[0]
        reg_intensity, reg_gradient = reg_profile_data[0]
        
        # Cari baris yang sama atau terdekat di adaptive sobel
        closest_row_idx = 0
        min_diff = float('inf')
        for i, row in enumerate(adap_profile_rows):
            diff = abs(row - reg_row)
            if diff < min_diff:
                min_diff = diff
                closest_row_idx = i
        
        adap_row = adap_profile_rows[closest_row_idx]
        adap_intensity, adap_gradient = adap_profile_data[closest_row_idx]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(reg_gradient, 'r-', label='Gradien Sobel Biasa')
        ax.plot(adap_gradient, 'g-', label='Gradien Adaptive Sobel')
        
        # Tandai piksel tepi
        reg_edge_positions = np.where(regular_sobel[reg_row, :] > 0)[0]
        adap_edge_positions = np.where(adaptive_sobel[adap_row, :] > 0)[0]
        
        if len(reg_edge_positions) > 0:
            ax.plot(reg_edge_positions, reg_gradient[reg_edge_positions], 'ro', label='Tepi Sobel Biasa')
        
        if len(adap_edge_positions) > 0:
            ax.plot(adap_edge_positions, adap_gradient[adap_edge_positions], 'go', label='Tepi Adaptive Sobel')
        
        ax.set_title(f'Perbandingan Gradien (Baris ~{reg_row})')
        ax.set_xlabel('Posisi Piksel')
        ax.set_ylabel('Nilai Gradien')
        ax.legend()
        
        st.pyplot(fig)
        
        # Tambahkan penjelasan interpretasi
        st.write("""
        ### Interpretasi Grafik Profil Kontras:
        
        1. **Puncak Gradien**: Puncak yang lebih tinggi menunjukkan perubahan intensitas yang lebih tajam, yang biasanya merupakan tepi yang kuat.
        
        2. **Sensitivitas**: Metode dengan kemampuan mendeteksi puncak-puncak kecil menunjukkan sensitivitas yang lebih tinggi terhadap perubahan kontras yang halus.
        
        3. **Konsistensi**: Perhatikan apakah metode tersebut konsisten dalam mendeteksi tepi di seluruh gambar, atau hanya pada area dengan kontras tinggi saja.
        
        4. **Noise**: Puncak-puncak kecil yang tidak beraturan mungkin menunjukkan noise, bukan tepi yang sebenarnya.
        
        Adaptive Sobel umumnya menunjukkan sensitivitas yang lebih baik pada area dengan kontras rendah, sementara tetap dapat mendeteksi tepi yang kuat pada area dengan kontras tinggi.
        """)
    # NEW SECTION: Edge Map Overlay
    st.header("Overlay Tepi pada Gambar Asli")
    st.write("""
    Overlay tepi adalah representasi visual langsung dari tepi yang terdeteksi pada gambar asli. 
    Ini memungkinkan kita untuk mengevaluasi dengan jelas apakah tepi yang terdeteksi sesuai dengan objek asli atau tidak.
    
    **Interpretasi:**
    - **Overlay yang baik:** Jika overlay tepi menunjukkan bahwa sebagian besar tepi terdeteksi dengan benar, maka deteksi tepi tersebut berhasil dan akurat.
    - **Overlay yang buruk:** Jika overlay menunjukkan banyak false positives (FP) atau false negatives (FN), artinya ada kesalahan dalam mendeteksi tepi.
    """)
    
    # Create edge overlays with different colors for better distinction
    regular_overlay = create_edge_overlay(image, regular_sobel, color=(255, 0, 0))  # Merah untuk Sobel Biasa
    adaptive_overlay = create_edge_overlay(image, adaptive_sobel, color=(0, 255, 0))  # Hijau untuk Adaptive Sobel
    
    # Tampilkan overlay secara berdampingan
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Overlay Tepi - Sobel Biasa")
        st.image(cv2.cvtColor(regular_overlay, cv2.COLOR_BGR2RGB), caption="Tepi Sobel Biasa (Merah)", use_container_width=True)
        
        # Tambahkan penjelasan interpretasi
        st.write("""
        **Analisis Overlay Sobel Biasa:**
        - Perhatikan area dengan kontras rendah - apakah tepi terdeteksi dengan baik?
        - Perhatikan apakah ada false positives (tepi yang terdeteksi padahal seharusnya tidak ada)
        - Perhatikan apakah ada false negatives (tepi yang tidak terdeteksi padahal seharusnya ada)
        """)
    
    with col2:
        st.subheader("Overlay Tepi - Adaptive Sobel")
        st.image(cv2.cvtColor(adaptive_overlay, cv2.COLOR_BGR2RGB), caption="Tepi Adaptive Sobel (Hijau)", use_container_width=True)
        
        # Tambahkan penjelasan interpretasi
        st.write("""
        **Analisis Overlay Adaptive Sobel:**
        - Perhatikan area dengan kontras rendah - apakah tepi terdeteksi lebih baik dibandingkan Sobel Biasa?
        - Perhatikan apakah ada pengurangan false positives
        - Perhatikan apakah ada pengurangan false negatives
        """)
    
    # Tambahkan perbandingan langsung (overlay gabungan)
    st.subheader("Perbandingan Langsung Overlay Tepi")
    st.write("""
    Pada gambar di bawah ini, tepi dari kedua metode ditampilkan bersamaan pada gambar asli:
    - **Merah**: Tepi yang terdeteksi oleh Sobel Biasa
    - **Hijau**: Tepi yang terdeteksi oleh Adaptive Sobel
    - **Kuning**: Area di mana kedua metode mendeteksi tepi yang sama
    
    Perhatikan perbedaan dan kesamaan antara kedua metode deteksi tepi.
    """)
    
    # Buat overlay gabungan
    combined_overlay = image.copy()
    if len(combined_overlay.shape) == 2:
        combined_overlay = cv2.cvtColor(combined_overlay, cv2.COLOR_GRAY2BGR)
    
    # Mask untuk tepi Sobel Biasa (merah)
    regular_mask = regular_sobel > 0
    # Mask untuk tepi Adaptive Sobel (hijau)
    adaptive_mask = adaptive_sobel > 0
    # Mask untuk area yang terdeteksi oleh kedua metode (kuning)
    both_mask = regular_mask & adaptive_mask
    
    # Terapkan warna
    combined_overlay[regular_mask & ~both_mask] = [0, 0, 255]  # Merah untuk Sobel Biasa saja
    combined_overlay[adaptive_mask & ~both_mask] = [0, 255, 0]  # Hijau untuk Adaptive Sobel saja
    combined_overlay[both_mask] = [0, 255, 255]  # Kuning untuk kedua metode
    
    # Tampilkan overlay gabungan
    st.image(cv2.cvtColor(combined_overlay, cv2.COLOR_BGR2RGB), caption="Perbandingan Overlay Tepi (Merah: Sobel Biasa, Hijau: Adaptive Sobel, Kuning: Keduanya)", use_container_width=True)
    
    # Tambahkan penjelasan interpretasi perbandingan
    st.write("""
    **Interpretasi Perbandingan Overlay:**
    
    1. **Area Merah (Sobel Biasa saja)**: 
       - Ini menunjukkan tepi yang hanya terdeteksi oleh Sobel Biasa
       - Jika banyak area merah yang tidak sesuai dengan tepi sebenarnya, ini menunjukkan false positives dari Sobel Biasa
    
    2. **Area Hijau (Adaptive Sobel saja)**:
       - Ini menunjukkan tepi yang hanya terdeteksi oleh Adaptive Sobel
       - Jika area hijau lebih sesuai dengan tepi sebenarnya, ini menunjukkan keunggulan Adaptive Sobel
    
    3. **Area Kuning (Kedua metode)**:
       - Ini menunjukkan tepi yang terdeteksi oleh kedua metode
       - Area kuning yang banyak pada tepi yang jelas menunjukkan bahwa kedua metode bekerja baik pada tepi dengan kontras tinggi
    
    Adaptive Sobel umumnya menunjukkan performa yang lebih baik pada area dengan kontras rendah, sementara kedua metode cenderung sama pada area dengan kontras tinggi.
    """)
    
    # Tambahkan metrik kuantitatif untuk overlay
    st.subheader("Metrik Kuantitatif Overlay")
    
    # Hitung jumlah piksel untuk masing-masing kategori
    regular_only_count = np.sum(regular_mask & ~adaptive_mask)
    adaptive_only_count = np.sum(adaptive_mask & ~regular_mask)
    both_count = np.sum(both_mask)
    
    # Hitung total piksel tepi
    total_regular = np.sum(regular_mask)
    total_adaptive = np.sum(adaptive_mask)
    
    # Tampilkan metrik dalam bentuk tabel
    overlay_metrics = {
        "Metrik": ["Jumlah Piksel Tepi", "Tepi Unik", "Tepi Bersama", "% Tepi Unik", "% Tepi Bersama"],
        "Sobel Biasa": [
            total_regular,
            regular_only_count,
            both_count,
            f"{(regular_only_count / total_regular * 100):.2f}%" if total_regular > 0 else "N/A",
            f"{(both_count / total_regular * 100):.2f}%" if total_regular > 0 else "N/A"
        ],
        "Adaptive Sobel": [
            total_adaptive,
            adaptive_only_count,
            both_count,
            f"{(adaptive_only_count / total_adaptive * 100):.2f}%" if total_adaptive > 0 else "N/A",
            f"{(both_count / total_adaptive * 100):.2f}%" if total_adaptive > 0 else "N/A"
        ]
    }
    
    st.table(overlay_metrics)
    
    st.write("""
    **Interpretasi Metrik Overlay:**
    
    - **Jumlah Piksel Tepi**: Total piksel yang terdeteksi sebagai tepi oleh masing-masing metode
    - **Tepi Unik**: Piksel tepi yang hanya terdeteksi oleh satu metode saja
    - **Tepi Bersama**: Piksel tepi yang terdeteksi oleh kedua metode
    - **% Tepi Unik**: Persentase tepi yang hanya terdeteksi oleh satu metode
    - **% Tepi Bersama**: Persentase tepi yang terdeteksi oleh kedua metode
    
    Persentase tepi bersama yang tinggi menunjukkan bahwa kedua metode memiliki kemiripan dalam mendeteksi tepi. Persentase tepi unik yang tinggi menunjukkan perbedaan signifikan antara kedua metode.
    """)
    
    # Hitung dan tampilkan metrik evaluasi dasar
    st.header("Metrik Evaluasi Dasar")
    
    # Calculate basic metrics (without ground truth)
    metrics_regular = calculate_metrics(gray_original, regular_sobel)
    metrics_adaptive = calculate_metrics(gray_original, adaptive_sobel)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Metrik Sobel Biasa")
        st.write(f"Jumlah Piksel Tepi: {metrics_regular['Edge Pixel Count']}")
        st.write(f"PSNR: {metrics_regular['PSNR'] if isinstance(metrics_regular['PSNR'], str) else metrics_regular['PSNR']:.2f}")
        st.write(f"SSIM: {metrics_regular['SSIM'] if isinstance(metrics_regular['SSIM'], str) else metrics_regular['SSIM']:.2f}")
    
    with col2:
        st.subheader("Metrik Adaptive Sobel")
        st.write(f"Jumlah Piksel Tepi: {metrics_adaptive['Edge Pixel Count']}")
        st.write(f"PSNR: {metrics_adaptive['PSNR'] if isinstance(metrics_adaptive['PSNR'], str) else metrics_adaptive['PSNR']:.2f}")
        st.write(f"SSIM: {metrics_adaptive['SSIM'] if isinstance(metrics_adaptive['SSIM'], str) else metrics_adaptive['SSIM']:.2f}")
    
    # Perbandingan langsung
    st.header("Analisis Perbandingan")
    
    # Edge count comparison
    edge_diff = metrics_adaptive["Edge Pixel Count"] - metrics_regular["Edge Pixel Count"]
    edge_diff_percent = (edge_diff / metrics_regular["Edge Pixel Count"] * 100) if metrics_regular["Edge Pixel Count"] > 0 else 0
    
    st.write(f"Perbedaan Jumlah Piksel Tepi: {edge_diff} piksel ({edge_diff_percent:.2f}%)")
    
    # Contrast comparison
    contrast_diff = adap_avg_contrast - reg_avg_contrast
    contrast_diff_percent = (contrast_diff / reg_avg_contrast * 100) if reg_avg_contrast > 0 else 0
    
    st.write(f"Perbedaan Rata-rata Kontras Lokal: {contrast_diff:.2f} ({contrast_diff_percent:.2f}%)")
    
    # Intensity comparison
    intensity_diff = adap_mean - reg_mean
    intensity_diff_percent = (intensity_diff / reg_mean * 100) if reg_mean > 0 else 0
    
    st.write(f"Perbedaan Rata-rata Intensitas Tepi: {intensity_diff:.2f} ({intensity_diff_percent:.2f}%)")
    
    # Tampilkan kesimpulan berdasarkan metrik
    st.subheader("Kesimpulan")
    
    conclusions = []
    
    if metrics_adaptive["Edge Pixel Count"] > metrics_regular["Edge Pixel Count"]:
        conclusions.append("- Adaptive Sobel mendeteksi lebih banyak tepi dibandingkan Sobel biasa, menunjukkan sensitivitas yang lebih tinggi terhadap perubahan intensitas lokal.")
    else:
        conclusions.append("- Sobel biasa mendeteksi lebih banyak tepi dibandingkan Adaptive Sobel, yang mungkin mengindikasikan lebih banyak noise terdeteksi.")
    
    if adap_avg_contrast > reg_avg_contrast:
        conclusions.append("- Adaptive Sobel mendeteksi tepi pada area dengan kontras lokal yang lebih tinggi, menunjukkan kemampuan yang lebih baik dalam mendeteksi tepi yang signifikan.")
    else:
        conclusions.append("- Sobel biasa mendeteksi tepi pada area dengan kontras lokal yang lebih tinggi.")
    
    if adap_std < reg_std:
        conclusions.append("- Adaptive Sobel menghasilkan intensitas tepi yang lebih konsisten (standar deviasi lebih rendah), menunjukkan deteksi tepi yang lebih stabil.")
    else:
        conclusions.append("- Sobel biasa menghasilkan intensitas tepi yang lebih konsisten.")
    
    for conclusion in conclusions:
        st.write(conclusion)
    
    st.write("""
    **Interpretasi Metrik Baru:**
    
    **1. Distribusi Intensitas Tepi**: 
    Menunjukkan bagaimana intensitas piksel tepi terdistribusi. Distribusi yang lebih merata dengan standar deviasi yang lebih rendah menunjukkan deteksi tepi yang lebih konsisten.
    
    **2. Sensitivitas Kontras Lokal**: 
    Mengukur seberapa baik metode deteksi tepi dalam merespons perubahan kontras lokal. Nilai kontras lokal yang lebih tinggi menunjukkan bahwa metode tersebut lebih baik dalam mendeteksi tepi pada area dengan variasi intensitas yang signifikan.
    
    **3. Overlay Tepi**: 
    Visualisasi langsung dari tepi yang terdeteksi pada gambar asli. Memungkinkan evaluasi visual terhadap akurasi dan kelengkapan deteksi tepi.
    
    **Keunggulan Adaptive Sobel:**
    Adaptive Sobel umumnya lebih unggul dalam mendeteksi tepi pada gambar dengan variasi pencahayaan dan kontras yang tidak merata. Metode ini menyesuaikan threshold berdasarkan karakteristik lokal gambar, sehingga dapat mendeteksi tepi dengan lebih baik pada area dengan kontras rendah maupun tinggi.
    """)
else:
    st.info("Silakan pilih gambar untuk memulai perbandingan.")