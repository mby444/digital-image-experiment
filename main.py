import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURASI ---
INPUT_DIR = 'input_images'
OUTPUT_DIR = 'output_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_file_size(file_path):
    """Mengambil ukuran file dalam KB."""
    return os.path.getsize(file_path) / 1024

def calculate_psnr(orig, processed):
    """Menghitung PSNR. Jika ukuran beda, processed di-upscale ke ukuran orig."""
    h, w = orig.shape[:2]
    if processed.shape[:2] != (h, w):
        processed = cv2.resize(processed, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Pastikan jumlah channel sama untuk perhitungan (konversi ke gray jika perlu)
    if len(orig.shape) == 3 and (len(processed.shape) == 2 or processed.shape[2] == 1):
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        return cv2.PSNR(orig_gray, processed)
    
    return cv2.PSNR(orig, processed)

def create_combined_plot(data, title, filename):
    """
    Membuat grafik perbandingan ganda: 
    1. Bar Chart untuk Ukuran File (KB)
    2. Line Chart untuk Kualitas Citra (PSNR dalam dB)
    
    Parameter:
    data (dict): Dictionary berisi kategori dan data citra (sizes & psnrs).
    title (str): Judul utama grafik.
    filename (str): Nama file untuk menyimpan hasil plot.
    """
    labels = data['categories']
    images = data['images']
    
    # Menambah tinggi figure agar label tidak terpotong (12 -> 14)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    x = np.arange(len(labels))
    width = 0.2
    
    # 1. Bar Chart (Ukuran File) - Tidak berubah banyak
    for i, img_data in enumerate(images):
        # Mengatur posisi batang agar tidak tumpang tindih
        offset = (i - 1) * width
        rects = ax1.bar(x + offset, img_data['sizes'], width, label=img_data['name'])
        
        # Menambahkan label angka presisi di atas setiap batang
        ax1.bar_label(rects, padding=3, fmt='%.1f', fontsize=9)
    
    ax1.set_ylabel('Ukuran File (KB)')
    ax1.set_title(f'{title} - Perbandingan Ukuran')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # --- BAGIAN 2: GRAFIK KUALITAS CITRA / PSNR (LINE CHART) ---
    for i, img_data in enumerate(images):
        # Plot garis dan ambil warnanya
        line = ax2.plot(labels, img_data['psnrs'], marker='o', label=img_data['name'], linewidth=2.5)
        color = line[0].get_color()
        
        # STRATEGI: Staggered Offset (i=0 -> 10, i=1 -> 25, i=2 -> 40)
        # Menumpuk label secara vertikal agar tidak bertabrakan di titik X yang sama
        y_label_offset = 10 + (i * 15) 
        
        for j, val in enumerate(img_data['psnrs']):
            ax2.annotate(f'{val:.1f}', 
                         (labels[j], img_data['psnrs'][j]),
                         textcoords="offset points", 
                         xytext=(0, y_label_offset), 
                         ha='center', 
                         fontsize=10,
                         fontweight='bold',
                         color=color, # Warna teks sesuai warna garis
                         bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8, ec=color))
    
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title(f'{title} - Perbandingan Kualitas (PSNR)')
    
    # Beri margin atas yang lebih luas (50 dB ekstra) agar label tertinggi tidak terpotong
    if images:
        max_psnr = max([max(img['psnrs']) for img in images])
        ax2.set_ylim(0, max_psnr + 60) 
        
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    print(f"Grafik diperbarui: {filename}")

def process_assignment():
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) < 3:
        print("Error: Masukkan minimal 3 gambar di folder 'input_images'.")
        return

    # Struktur Data untuk Plotting
    results = {
        'resize': {'categories': ['Original', '50%', '25%'], 'images': []},
        'color': {'categories': ['Grayscale', '16-Gray', 'Biner'], 'images': []},
        'interpolation': {'categories': ['Nearest', 'Bilinear', 'Bicubic'], 'images': []}
    }

    for file_name in image_files[:3]:
        path = os.path.join(INPUT_DIR, file_name)
        img = cv2.imread(path)
        if img is None: continue
        
        # --- 1. DATA RESIZE ---
        res_sizes, res_psnrs = [get_file_size(path)], [50.0]
        for sc in [0.5, 0.25]:
            tmp = cv2.resize(img, None, fx=sc, fy=sc)
            p = os.path.join(OUTPUT_DIR, f"resize_{int(sc*100)}_{file_name}")
            cv2.imwrite(p, tmp)
            res_sizes.append(get_file_size(p))
            res_psnrs.append(calculate_psnr(img, tmp))
        results['resize']['images'].append({'name': file_name, 'sizes': res_sizes, 'psnrs': res_psnrs})

        # --- 2. DATA WARNA/KUANTISASI ---
        # Untuk warna, kita bandingkan dengan versi grayscale original agar adil
        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c_sizes, c_psnrs = [], []
        
        # Grayscale
        p_g = os.path.join(OUTPUT_DIR, f"gray_{file_name}")
        cv2.imwrite(p_g, gray_orig)
        c_sizes.append(get_file_size(p_g)); c_psnrs.append(50.0)
        
        # 16 Graylevel
        g16 = (np.floor(gray_orig / 16) * 16).astype(np.uint8)
        p_16 = os.path.join(OUTPUT_DIR, f"gray16_{file_name}")
        cv2.imwrite(p_16, g16)
        c_sizes.append(get_file_size(p_16)); c_psnrs.append(calculate_psnr(gray_orig, g16))
        
        # Biner
        _, bin_img = cv2.threshold(gray_orig, 127, 255, cv2.THRESH_BINARY)
        p_b = os.path.join(OUTPUT_DIR, f"binary_{file_name}")
        cv2.imwrite(p_b, bin_img)
        c_sizes.append(get_file_size(p_b)); c_psnrs.append(calculate_psnr(gray_orig, bin_img))
        
        results['color']['images'].append({'name': file_name, 'sizes': c_sizes, 'psnrs': c_psnrs})

        # --- 3. DATA INTERPOLASI ---
        # Simulasi: Downscale ke 25% lalu Upscale kembali ke ukuran asli dengan 3 metode
        small = cv2.resize(img, None, fx=0.25, fy=0.25)
        h, w = img.shape[:2]
        i_sizes, i_psnrs = [], []
        
        for name, method in [("Nearest", cv2.INTER_NEAREST), ("Bilinear", cv2.INTER_LINEAR), ("Bicubic", cv2.INTER_CUBIC)]:
            inter = cv2.resize(small, (w, h), interpolation=method)
            p_i = os.path.join(OUTPUT_DIR, f"inter_{name}_{file_name}")
            cv2.imwrite(p_i, inter)
            i_sizes.append(get_file_size(p_i))
            i_psnrs.append(calculate_psnr(img, inter))
            
        results['interpolation']['images'].append({'name': file_name, 'sizes': i_sizes, 'psnrs': i_psnrs})
        print(f"Selesai memproses: {file_name}")

    # Generate 3 Grafik
    create_combined_plot(results['resize'], "Analisis Sampling (Resize)", "grafik_1_resize.png")
    create_combined_plot(results['color'], "Analisis Kuantisasi (Warna)", "grafik_2_warna.png")
    create_combined_plot(results['interpolation'], "Analisis Interpolasi", "grafik_3_interpolasi.png")
    
    print(f"\nSemua grafik telah disimpan di folder '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    process_assignment()