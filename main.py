import cv2
import os
import numpy as np

# Konfigurasi: Masukkan 3 gambar Anda ke dalam folder 'input_images'
INPUT_DIR = 'input_images'
OUTPUT_DIR = 'output_results'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def get_file_size(file_path):
    return os.path.getsize(file_path) / 1024  # Hasil dalam KB

def process_images():
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"{'File':<20} | {'Res Awal':<15} | {'Size (KB)':<10} | {'Format'}")
    print("-" * 60)

    for file_name in image_files[:3]: # Mengambil 3 gambar berbeda 
        path = os.path.join(INPUT_DIR, file_name)
        img = cv2.imread(path)
        
        if img is None: continue

        # 1 & 2. Catat metadata awal 
        h, w, c = img.shape
        f_size = get_file_size(path)
        f_ext = os.path.splitext(file_name)[1]
        print(f"{file_name:<20} | {w}x{h:<10} | {f_size:>8.2f} | {f_ext}")

        # 3. Resize 50% dan 25% [cite: 8]
        for scale in [0.5, 0.25]:
            resized = cv2.resize(img, None, fx=scale, fy=scale)
            out_name = f"{file_name}_resize_{int(scale*100)}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, resized)
            print(f"   -> Resized {int(scale*100)}%: {get_file_size(out_path):.2f} KB")

        # 4. Konversi Warna & Kuantisasi 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_grayscale.jpg"), gray)

        # Kuantisasi 16 Graylevel (Kuantisasi Iintensitas)
        gray16 = (np.floor(gray / 16) * 16).astype(np.uint8)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_gray16.jpg"), gray16)

        # Biner (Thresholding)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_binary.jpg"), binary)

        # 5. Perbandingan Interpolasi (Upscaling untuk melihat perbedaan) 
        # Kita perbesar kembali gambar yang sudah di-resize 25% ke ukuran asli
        small_img = cv2.resize(img, None, fx=0.25, fy=0.25)
        methods = [
            ("Nearest", cv2.INTER_NEAREST),
            ("Bilinear", cv2.INTER_LINEAR),
            ("Bicubic", cv2.INTER_CUBIC)
        ]
        
        for name, method in methods:
            inter_img = cv2.resize(small_img, (w, h), interpolation=method)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{file_name}_inter_{name}.jpg"), inter_img)

if __name__ == "__main__":
    process_images()