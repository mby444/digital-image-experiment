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
    """
    Menghitung kualitas citra menggunakan PSNR (Peak Signal-to-Noise Ratio).
    Semakin tinggi nilai dB, semakin mirip kualitasnya dengan gambar asli.
    """
    h, w = orig.shape[:2]
    # Upscale kembali gambar hasil resize ke ukuran asli agar bisa dibandingkan
    processed_up = cv2.resize(processed, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # Fungsi bawaan OpenCV untuk menghitung PSNR
    psnr = cv2.PSNR(orig, processed_up)
    return psnr

def process_assignment():
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) < 3:
        print("Peringatan: Pastikan ada minimal 3 gambar di folder 'input_images'.")
        return

    # Data untuk grafik
    plot_data = {
        'labels': ['Original', 'Resize 50%', 'Resize 25%'],
        'images_info': [] 
    }

    print(f"{'File':<15} | {'Size (KB)':<10} | {'PSNR (dB)':<10}")
    print("-" * 45)

    for file_name in image_files[:3]:
        path = os.path.join(INPUT_DIR, file_name)
        img = cv2.imread(path)
        if img is None: continue

        f_size_orig = get_file_size(path)
        # Benchmark kualitas gambar asli (ideal/sempurna)
        current_sizes = [f_size_orig]
        current_psnr = [50.0] # Representasi nilai maksimal untuk grafik

        # 3. Resize 50% dan 25% serta Hitung Kualitas
        for scale in [0.5, 0.25]:
            resized = cv2.resize(img, None, fx=scale, fy=scale)
            out_name = f"{int(scale*100)}_{file_name}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, resized)
            
            current_sizes.append(get_file_size(out_path))
            current_psnr.append(calculate_psnr(img, resized))

        plot_data['images_info'].append({
            'name': file_name, 
            'sizes': current_sizes, 
            'psnr': current_psnr
        })

        # 4. Konversi Warna & Kuantisasi (Tetap diproses untuk file hasil)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"gray_{file_name}"), gray)
        gray16 = (np.floor(gray / 16) * 16).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"gray16_{file_name}"), gray16)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"binary_{file_name}"), binary)
        
        print(f"{file_name:<15} | {f_size_orig:>9.1f} | {'Asli'}")
        print(f"{'  (Resize 50%)':<15} | {current_sizes[1]:>9.1f} | {current_psnr[1]:>9.2f}")
        print(f"{'  (Resize 25%)':<15} | {current_sizes[2]:>9.1f} | {current_psnr[2]:>9.2f}")

    # Buat grafik perbandingan ganda
    create_dual_plot(plot_data)

def create_dual_plot(data):
    labels = data['labels']
    info = data['images_info']
    x = np.arange(len(labels))
    width = 0.2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Grafik 1: Perbandingan Ukuran File (Bar Chart)
    for i, item in enumerate(info):
        ax1.bar(x + (i-1)*width, item['sizes'], width, label=item['name'])
    ax1.set_ylabel('Ukuran File (KB)')
    ax1.set_title('Perbandingan Efisiensi Penyimpanan (Ukuran File)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()

    # Grafik 2: Perbandingan Kualitas (Line Chart - PSNR)
    for i, item in enumerate(info):
        ax2.plot(labels, item['psnr'], marker='o', label=item['name'], linewidth=2)
    ax2.set_ylabel('PSNR (dB) - Lebih tinggi lebih baik')
    ax2.set_title('Perbandingan Kualitas Citra (PSNR)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'analisis_kualitas_dan_ukuran.png'))
    print(f"\nSelesai! Grafik analisis disimpan di: {OUTPUT_DIR}/analisis_kualitas_dan_ukuran.png")

if __name__ == "__main__":
    process_assignment()