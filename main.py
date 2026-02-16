import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURASI ---
INPUT_DIR = 'input_images'   # Folder berisi 3 gambar (Foto HP, Screenshot, Internet)
OUTPUT_DIR = 'output_results' # Folder hasil eksperimen

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_file_size(file_path):
    """Mengambil ukuran file dalam satuan KB."""
    return os.path.getsize(file_path) / 1024

def process_assignment():
    # Mengambil list file gambar
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) < 3:
        print("Peringatan: Pastikan ada minimal 3 gambar di folder 'input_images' sesuai instruksi.")
        return

    # Data untuk visualisasi grafik
    plot_data = {
        'labels': ['Original', 'Resize 50%', 'Resize 25%'],
        'images_info': [] # Menyimpan {name: str, sizes: [orig, r50, r25]}
    }

    print(f"{'File':<20} | {'Resolusi':<15} | {'Size (KB)':<10} | {'Format'}")
    print("-" * 65)

    for file_name in image_files[:3]: # Memproses 3 gambar pertama 
        path = os.path.join(INPUT_DIR, file_name)
        img = cv2.imread(path)
        
        if img is None: continue

        # 1 & 2. Catat metadata awal 
        h, w, c = img.shape
        f_size_orig = get_file_size(path)
        f_ext = os.path.splitext(file_name)[1]
        print(f"{file_name:<20} | {w}x{h:<10} | {f_size_orig:>8.2f} | {f_ext}")

        current_image_sizes = [f_size_orig]

        # 3. Resize 50% dan 25% serta catat ukuran file [cite: 8]
        for scale in [0.5, 0.25]:
            resized = cv2.resize(img, None, fx=scale, fy=scale)
            out_name = f"res_{int(scale*100)}_{file_name}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, resized)
            current_image_sizes.append(get_file_size(out_path))

        plot_data['images_info'].append({'name': file_name, 'sizes': current_image_sizes})

        # 4. Konversi Warna & Kuantisasi [cite: 9]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"gray_{file_name}"), gray)

        # 16 Graylevel (Kuantisasi Intensitas)
        gray16 = (np.floor(gray / 16) * 16).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"gray16_{file_name}"), gray16)

        # Biner (Thresholding)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"binary_{file_name}"), binary)

        # 5. Interpolasi (Upscaling untuk melihat perbedaan kualitas) [cite: 9]
        # Memperbesar kembali gambar yang sudah di-resize 25% ke ukuran asli
        small_img = cv2.resize(img, None, fx=0.25, fy=0.25)
        methods = [
            ("Nearest", cv2.INTER_NEAREST),
            ("Bilinear", cv2.INTER_LINEAR),
            ("Bicubic", cv2.INTER_CUBIC)
        ]
        
        for name, method in methods:
            inter_img = cv2.resize(small_img, (w, h), interpolation=method)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"inter_{name}_{file_name}"), inter_img)

    # --- GENERASI GRAFIK --- [cite: 19]
    create_comparison_plot(plot_data)
    print(f"\nEksperimen selesai! Hasil gambar dan grafik ada di folder '{OUTPUT_DIR}'.")

def create_comparison_plot(data):
    labels = data['labels']
    info = data['images_info']
    
    x = np.arange(len(labels))
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = ['#155bd5', '#d98d00', '#009664']
    for i, item in enumerate(info):
        offset = (i - 1) * width
        rects = ax.bar(x + offset, item['sizes'], width, label=item['name'], color=colors[i % 3])
        ax.bar_label(rects, padding=3, fmt='%.1f', fontsize=8)

    ax.set_ylabel('Ukuran File (KB)')
    ax.set_title('Perbandingan Ukuran File Citra: Original vs Resize')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'grafik_perbandingan.png'))

if __name__ == "__main__":
    process_assignment()