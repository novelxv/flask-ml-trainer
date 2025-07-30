# Instruksi Penggunaan Flask ML Trainer

## 🚀 Cara Menjalankan Aplikasi

### Opsi 1: Jalankan Langsung (Termudah)
1. Double-click file `run.bat`
2. Tunggu hingga aplikasi terbuka di browser
3. Jika tidak terbuka otomatis, buka browser dan ke: http://localhost:5000

### Opsi 2: Membuat File EXE
1. Double-click file `build.bat`
2. Tunggu proses building selesai
3. File executable akan tersedia di folder `dist\FlaskMLTrainer.exe`
4. Double-click `FlaskMLTrainer.exe` untuk menjalankan

## 📊 Cara Menggunakan Aplikasi

### 1. Upload Dataset
- Klik "Upload Dataset" di halaman utama
- Pilih file Excel (.xlsx, .xls) atau CSV (.csv)
- Atau gunakan "Load Iris Dataset" untuk contoh

### 2. Konfigurasi Training
- Pilih kolom target (yang ingin diprediksi)
- Atur parameter training:
  - Test size: porsi data untuk testing (default: 0.2)
  - Number of trees: jumlah pohon dalam model (default: 100)
  - Max depth: kedalaman maksimum pohon (opsional)

### 3. Training Model
- Klik "Train Model"
- Tunggu proses training selesai
- Lihat hasil akurasi dan laporan klasifikasi

### 4. Melakukan Prediksi
- Klik "Make Predictions" di menu atau hasil training
- Masukkan nilai untuk setiap fitur
- Klik "Predict"
- Lihat hasil prediksi dengan confidence score dan probabilitas

## 📁 Struktur File

```
flask-ml-trainer/
├── app.py              # Aplikasi utama Flask
├── start_app.py        # Starter script dengan auto-browser
├── requirements.txt    # Dependencies Python
├── app.spec           # Konfigurasi PyInstaller
├── build.bat          # Script build executable
├── run.bat            # Script jalankan development
├── templates/         # Template HTML
├── static/           # File CSS
├── uploads/          # Dataset yang diupload (dibuat otomatis)
└── models/           # Model yang disimpan (dibuat otomatis)
```

## 🔧 Requirements

- Python 3.7 atau lebih baru
- Windows 10/11 (untuk build executable)
- RAM minimal 4GB
- Disk space 500MB

## 🚨 Troubleshooting

### Error "No module named..."
- Jalankan: `pip install -r requirements.txt`

### Aplikasi tidak terbuka di browser
- Buka manual: http://localhost:5000

### Error saat building executable
- Pastikan antivirus tidak memblokir
- Jalankan sebagai administrator

### Model accuracy rendah
- Pastikan dataset cukup besar (minimal 50+ samples)
- Cek kolom target sudah benar
- Coba adjust parameter training

## 📞 Fitur Utama

✅ Upload Excel/CSV datasets
✅ Preview data interaktif
✅ Training otomatis dengan Random Forest
✅ Prediksi real-time
✅ Confidence score dan probabilitas
✅ Interface yang cantik dan responsif
✅ Sample dataset Iris
✅ Export ke executable (.exe)
✅ Model persistence (otomatis save/load)

## 💡 Tips

- Gunakan dataset dengan minimal 4-5 kolom untuk hasil terbaik
- Untuk dataset besar, increase number of trees
- Test size 0.2-0.3 biasanya optimal
- Gunakan sample dataset Iris untuk belajar dulu

Selamat menggunakan Flask ML Trainer! 🤖
