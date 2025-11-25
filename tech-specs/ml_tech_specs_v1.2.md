# Panduan Teknis Model Machine Learning: Sistem Klasifikasi Body Condition Score (BCS) Sapi (v1.2)
**Versi 1.2 — Integrasi YOLOv8 Classification pada Dataset Raw**

---

## Executive Summary

- **Fokus v1.2**: Penyesuaian spesifikasi untuk memanfaatkan struktur dataset yang tersedia di `data/raw` (terkelompok berdasarkan folder skor) menggunakan arsitektur **YOLOv8-CLS (Classification)**.
- **Data**: Dataset lokal pada folder `data/raw` yang terorganisir dalam sub-folder kelas skor (3.25, 3.5, 3.75, 4.0, 4.25).
- **Model**: **YOLOv8-CLS** (Ultralytics) untuk klasifikasi langsung skor BCS. Pendekatan ini dipilih karena kesesuaian langsung dengan struktur folder dataset tanpa memerlukan anotasi bounding box/masker tambahan di tahap awal.
- **Target**: Akurasi Top-1 > 70%, Top-2 > 90% (mengingat kemiripan visual antar kelas skor yang berdekatan).
- **Deployment**: API FastAPI ringan yang memuat model YOLOv8 format `.pt` atau `.onnx`.

---

## 1) Business Understanding & Penyesuaian Scope

### 1.1 Latar Belakang
Berdasarkan ketersediaan data di `data/raw` yang sudah terlabel dalam folder-folder skor, pendekatan paling efisien adalah memperlakukan masalah ini sebagai **Klasifikasi Gambar (Image Classification)** diskrit, alih-alih regresi murni atau segmentasi kompleks.

### 1.2 Tujuan
- Melatih model YOLOv8-CLS untuk memprediksi kelas BCS sapi.
- Mengotomasi pipeline dari `data/raw` ke format siap latih YOLO.
- Menyediakan API inferensi yang cepat.

### 1.3 Metrik Sukses
- **Akurasi (Top-1)**: Persentase prediksi tepat sesuai label folder.
- **Akurasi (Top-2)**: Persentase label benar berada di 2 prediksi teratas (penting karena beda 0.25 sangat tipis).
- **Confusion Matrix**: Memastikan error tidak melenceng jauh (misal: label 3.25 diprediksi 4.25 adalah error fatal, tapi diprediksi 3.5 masih dapat diterima).

---

## 2) Data (Struktur & Pipeline)

### 2.1 Struktur Data Saat Ini (`data/raw`)
Dataset tersimpan dengan struktur *ImageFolder* standar:
```text
bcs-cow-project/
└── ml/
    └── data/
        └── raw/
            ├── 3.25/      # Citra sapi dengan BCS 3.25
            ├── 3.5/       # Citra sapi dengan BCS 3.5
            ├── 3.75/      # Citra sapi dengan BCS 3.75
            ├── 4.0/       # Citra sapi dengan BCS 4.0
            └── 4.25/      # Citra sapi dengan BCS 4.25
```

### 2.2 Pipeline Preprocessing (Raw to Processed)
YOLOv8-CLS membutuhkan struktur direktori `train`, `val`, dan `test` yang masing-masing memiliki sub-folder kelas.

**Rencana Transformasi**:
Script `src/data/make_dataset.py` (akan dibuat) bertugas:
1. Membaca semua gambar dari `data/raw`.
2. Melakukan splitting (misal: 70% Train, 20% Val, 10% Test) secara acak namun *stratified* (mempertahankan proporsi tiap kelas).
3. Menyalin/memindahkan file ke struktur `data/processed`:
   ```text
   data/processed/
   ├── train/
   │   ├── 3.25/
   │   ├── 3.5/
   │   └── ...
   ├── val/
   │   ├── 3.25/
   │   └── ...
   └── test/
       ├── 3.25/
       └── ...
   ```

### 2.3 Augmentasi
Karena jumlah data mungkin terbatas, augmentasi diterapkan *on-the-fly* oleh YOLOv8 saat training:
- **Mosaic**: Off (kurang cocok untuk klasifikasi BCS yang butuh struktur tubuh utuh).
- **Scale/Crop**: Random crop ringan (0.8-1.0).
- **Flip**: Horizontal flip (kiri-kanan sapi simetris).
- **Color**: HSV augmentation ringan untuk variasi pencahayaan.

---

## 3) Desain Model (YOLOv8-CLS)

### 3.1 Arsitektur
Menggunakan **YOLOv8-Classification** (e.g., `yolov8n-cls.pt` atau `yolov8s-cls.pt`).
- **Backbone**: CSPDarknet (ekstraksi fitur visual kuat).
- **Head**: Classification Head (Softmax output sejumlah kelas: 5 kelas [3.25, 3.5, 3.75, 4.0, 4.25]).

### 3.2 Konfigurasi Training
Contoh konfigurasi (`config.yaml` atau argumen CLI):
```python
from ultralytics import YOLO

# Load model pre-trained
model = YOLO('yolov8n-cls.pt') 

# Train
results = model.train(
    data='ml/data/processed', # Path ke folder yang berisi train/val
    epochs=50,
    imgsz=224,                # Ukuran standar klasifikasi
    batch=16,
    project='bcs_project',
    name='yolov8n_bcs_v1',
    patience=10               # Early stopping
)
```

### 3.3 Kelebihan Pendekatan Ini
- **Sesuai Data**: Langsung memetakan folder `raw` ke training tanpa perlu labeling bounding box manual.
- **Cepat**: YOLOv8-CLS sangat ringan dan cepat untuk inferensi.
- **Baseline Kuat**: Jika akurasi kurang, bisa beralih ke pendekatan segmentasi (v1.0) di masa depan dengan data yang sama (tapi perlu labeling tambahan).

---

## 4) Evaluasi

### 4.1 Skenario Pengujian
Evaluasi dilakukan pada `data/processed/test`.

### 4.2 Metrik Utama
1. **Accuracy**: Akurasi keseluruhan.
2. **Class-wise Accuracy**: Akurasi per skor BCS (apakah model lemah di skor ekstrem atau tengah?).
3. **Inference Time**: Waktu rata-rata per gambar (target < 100ms).

---

## 5) Deployment (API)

### 5.1 Spesifikasi API (FastAPI)
Sama dengan v1.1, namun memuat model YOLO.

**Endpoint**: `POST /predict`
- **Input**: File gambar (JPG/PNG).
- **Proses**:
  1. Resize ke 224x224.
  2. Inferensi YOLOv8 (`model(image)`).
  3. Ambil kelas dengan probabilitas tertinggi (`probs.top1`).
- **Output**:
  ```json
  {
    "bcs_score": 3.5,
    "confidence": 0.85,
    "inference_time_ms": 45
  }
  ```

### 5.2 Dependensi
- `ultralytics`
- `fastapi`
- `uvicorn`
- `python-multipart`

---

## 6) Roadmap Implementasi (Next Steps)

1. **Data Splitting**: Buat script untuk membagi `data/raw` ke `data/processed`.
2. **Training**: Latih `yolov8n-cls` pada data processed.
3. **Evaluasi**: Cek hasil pada test set.
4. **API Dev**: Update `api/main.py` untuk menggunakan `ultralytics.YOLO`.
