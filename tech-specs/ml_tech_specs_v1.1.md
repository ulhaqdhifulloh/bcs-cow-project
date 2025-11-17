# Panduan Teknis Model Machine Learning: Sistem Klasifikasi Body Condition Score (BCS) Sapi
**Fase: Tahap Awal / Proof of Concept dengan Data Publik**

---

## Executive Summary

- **Fase saat ini**: Tahap awal menggunakan dataset publik (Dryad, UNICT/Penn Vet), model training lokal, fine-tuning pending setelah data lapangan tersedia.
- **Data**: Citra RGB publik (depth 16-bit disederhanakan ke RGB + ekstraksi fitur manual untuk PoC), atau RGB-only jika dataset depth publik terbatas.
- **Model**: Pipeline 2-tahap ringan — (1) CNN baseline (EfficientNet-B0 atau ResNet50 transfer learning) untuk ekstraksi fitur, (2) Regressor sederhana (MLPRegressor / XGBoost) untuk prediksi BCS.
- **Target tahap awal**: MAE < 0.40; R² > 0.65 (baseline); peningkatan ke MAE < 0.25 setelah fine-tuning dengan 10–20 sampel lapangan.
- **Inferensi**: API FastAPI sederhana di server (seperi Google Cloud / Azure / AWS atau lokal), terintegrasi dengan backend Laravel untuk menerima request.
- **Arsitektur**: Hybrid — mobile akuisisi (image collection), cloud inference (model & API), database MySQL menyimpan hasil prediksi.

---

## 1) Business Understanding

### 1.1 Latar Belakang & Masalah
- BCS penting untuk kesehatan reproduksi, produktivitas susu, dan manajemen bobot.
- Penilaian manual subjektif (variabilitas antarrater ±0.5–1.0), padat karya, dan berpotensi menimbulkan stres.
- Kebutuhan solusi **akurat, objektif, non-invasif**, dan **praktis** untuk peternak kecil–menengah; serta andal pada konektivitas terbatas.

### 1.2 Tujuan & Cakupan

**Tujuan Utama**:
- Mengotomasi estimasi BCS (skala 1.00–5.00, inkremen 0.25) dari citra RGB atau RGB-D.
- Mengembangkan pipeline ML end-to-end: *data preparation* → *model training* → *evaluation* → *API inference*.
- Menyediakan dashboard integrasi dengan sistem manajemen peternakan (Laravel backend).

**Cakupan Tahap Awal (Saat Ini)**:
- Eksperimen baseline dengan dataset publik.
- Pelatihan model lokal di Jupyter/Kaggle.
- API inferensi sederhana (FastAPI).
- Dokumentasi lengkap untuk fase berikutnya.

**Cakupan Tahap Berikutnya (Pending)**:
- Pengumpulan 10–20 sampel citra lapangan (fine-tuning).
- Optimisasi model berdasarkan data real.
- Deployment di production (Docker + cloud server).

**Di Luar Cakupan Awal**:
- Integrasi ERP peternakan kompleks.
- Timbangan otomatis atau sensor tambahan.
- Dukungan multi-spesies (saat ini hanya sapi).
- Model on-device di edge (dipertimbangkan fase lanjut).

### 1.3 Metrik Sukses

**Teknis (Tahap Awal)**:
- MAE < **0.40** (baseline PoC).
- R² > **0.65** (menjelaskan ≥65% varian target).
- Korelasi Pearson r > **0.75**.

**Teknis (Target Akhir, Setelah Fine-Tuning)**:
- MAE < **0.25**.
- R² > **0.75**.
- Korelasi r > **0.85**.

**Akurasi Operasional**:
- **Accuracy@±0.25** ≥ 90% (% prediksi dengan error ≤ 0.25 BCS).
- **Accuracy@±0.50** ≥ 95% (% prediksi dengan error ≤ 0.50 BCS).

**Sistem**:
- Latensi API < **5 detik** per prediksi (dari upload hingga respons).
- Uptime layanan ≥ **99.0%** (production).
- *Usability Score* (SUS) > **70** (dari feedback pengguna).

**Reliabilitas Data**:
- Inter-rater reliability (Cohen's κ tertimbang atau ICC) ≥ **0.75** (untuk tahap fine-tuning).

---

## 2) Data (Akuisisi, Struktur, Kualitas)

### 2.1 Sumber Data

**Tahap Awal (Saat Ini) — Data Publik**:

1. **Dryad Repository** — *Labeled RGB and depth images for cattle BCS*
   - Dataset dari Intel RealSense D435i.
   - Dimensi: RGB 8-bit + Depth 16-bit.
   - Label: BCS 1.00–5.00 (inkremen 0.25).
   - Jumlah: ~300–500 citra per studi.
   - URL: https://datadryad.org/ (cari "cattle BCS" atau "RealSense bovine").

2. **UNICT + Penn Vet BCS Database**
   - Citra RGB top-view sapi (format bergambar).
   - Anotasi anatomis & label BCS (visual reference).
   - Akses: Hubungi institusi atau cari publikasi terkait.

3. **Dataset Publik Alternatif**:
   - Citra dari Kaggle (jika ada dataset "cattle detection" atau "livestock monitoring").
   - ImageNet subset (bovine/livestock), untuk pre-training transfer learning.

**Tahap Berikutnya (Pending) — Data Lapangan**:

- **Lokasi**: Peternakan mitra (Jawa Barat), ~1–2 lokasi.
- **Target jumlah**: 10–20 sapi, masing-masing 5–10 foto (total ~50–100 citra).
- **Perangkat akuisisi**: 
  - Smartphone dengan kamera beresolusi ≥12MP (dapat menggunakan aplikasi sederhana untuk capture).
  - Action cam (GoPro, DJI Osmo, dll.) — optional, untuk backup/variasi sudut.
  - Deep camera (RealSense D435 / Kinect Azure) — opsional, jika memungkinkan, namun utamanya untuk uji implementasi di tahap lanjut.
- **Protokol**: Jarak 2–3 m, tinggi kamera 1–1.5 m, posisi lateral sapi, variasi pencahayaan.

### 2.2 Struktur Folder & Manajemen Berkas

```
bcs-project/
├── README.md                           # Overview proyek
├── requirements.txt                    # Dependensi Python
├── config.yaml                         # Konfigurasi (model params, path, dll)
│
├── data/                              # Data publik & internal
│   ├── raw/                           # Raw download dari sumber publik
│   │   ├── dryad_cattle_bcs/
│   │   │   ├── images_rgb/            # *.jpg / *.png (8-bit RGB)
│   │   │   ├── images_depth/          # *.png (16-bit, nilai dalam mm)
│   │   │   └── metadata.csv           # cow_id, bcs_label, source_device, dll
│   │   └── unict_penn_vet/
│   │       ├── images/
│   │       └── annotations.csv        # image_id, bcs_label, rater_name
│   │
│   ├── processed/                     # Data tersiap untuk ML
│   │   ├── train/
│   │   │   ├── images/                # Normalized RGB images
│   │   │   ├── features/              # Ekstraksi fitur (CSV atau pickle)
│   │   │   └── labels.csv             # cow_id, image_filename, bcs_label
│   │   ├── val/
│   │   │   ├── images/
│   │   │   ├── features/
│   │   │   └── labels.csv
│   │   └── test/
│   │       ├── images/
│   │       ├── features/
│   │       └── labels.csv
│   │
│   └── field_data/                    # Data lapangan (saat tersedia)
│       ├── cow_[01-20]/               # Per sapi
│       │   ├── *.jpg
│       │   ├── metadata.json
│       │   └── labels.csv
│
├── notebooks/                         # Jupyter notebook utama pipeline
│   └── 01_bcs_full_pipeline.ipynb    # SATU notebook: ringkasan business understanding, EDA, preprocessing, feature extraction, training, evaluasi & simpan/download model
│
├── models/                            # Model terlatih (artefak)
│   ├── cnn_backbone_v1.pt             # CNN pre-trained (transfer learning)
│   ├── bcs_regressor_v1.pkl           # Fitted regressor (scikit-learn/XGBoost)
│   └── model_metadata.json            # Version, training date, CV scores, dll
│
├── api/                               # FastAPI server (deployment, terpisah dari notebook)
│   ├── main.py                        # FastAPI app & routes
│   ├── inference.py                   # Logic prediksi (load model, forward pass)
│   ├── schemas.py                     # Pydantic models (request/response)
│   ├── database.py                    # Koneksi Laravel MySQL (async SQLAlchemy)
│   ├── config.py                      # Environment variables, settings
│   ├── requirements_api.txt           # Dependencies FastAPI
│   └── README.md                      # Dokumentasi atau panduan bagian API
│
├── mlflow/                            # MLflow tracking (opsional)
│   └── mlruns/                        # Eksperimen & metrik tercatat otomatis
│
├── tests/                             # Unit & integration tests
│   ├── test_data_loader.py
│   ├── test_inference.py
│   └── test_api.py
│
├── web/                               # Folder untuk projek web (tim web dev)
│
└── .gitignore                         # Exclude: data/, mlruns/, .cache, dll
```

### 2.3 Protokol Akuisisi Data (Tahap Lapangan)

**Perangkat**:
- Kamera: Smartphone (HP) dengan resolusi ≥12MP, fokus otomatis, stabilisasi gambar (opsional).
- Aksesori: Tripod kecil / penyangga untuk konsistensi posisi & tinggi.
- Penggaris / marker jarak untuk kalibrasi depth (jika menggunakan smartphone saja).

**Posisi & Jarak**:
- **Posisi sapi**: Berdiri alami, minimal stres, di lokasi dengan pencahayaan cukup.
- **Posisi kamera**: Lateral (samping), tinggi ~1.2–1.5 m, jarak 2–3 m dari tubuh sapi.
- **Sudut pengambilan**: Pastikan tubuh sapi terlihat utuh (dari bahu hingga panggul).
- Opsional: Ambil beberapa sudut (45°, 90°, 135°) untuk variasi, tapi minimal **konsisten per sapi**.

**Variasi Pencahayaan**:
- Pagi (6–9 pagi), Siang (10–14:00), Sore (15–17:30) atau minimalnya saat pencahayaan baik dan saat pencahaan buruk.
- Hindari backlight ekstrem (cahaya langsung di belakang sapi).

**Protokol per Sapi**:
1. Catat identitas sapi (ear tag, no. urut, nama, dll) → `cow_id`.
2. Ambil 5–10 foto per sapi di kondisi pencahayaan berbeda.
3. Pada form / metadata.json, catat:
   - `cow_id`, `timestamp`, `location`, `device_used` (HP/action cam).
   - Observasi kondisi sapi (sehat, gangguan, dll).
4. Upload ke folder `data/field_data/cow_[ID]/`.

**Labeling (Ground Truth)**:
- Gunakan **visual reference chart** (misal dari Penn Vet BCS).
- Minta ≥2 penilai independen (dokter hewan atau ahli ternak) untuk scoring.
- Label BCS: skala 1.00–5.00, inkremen 0.25.
- Simpan kedua skor; gunakan **rata-rata** sebagai label akhir.
- Catat perbedaan skornya → analisis inter-rater reliability (Cohen's κ atau ICC).

### 2.4 Data Split & Penghindaran Kebocoran (Leakage)

**Strategi Split** (untuk data publik tahap awal):
- **Train**: 70% data publik (digunakan untuk training & hyperparameter tuning).
- **Validation**: 15% data publik (digunakan untuk monitoring selama training, early stopping).
- **Test**: 15% data publik (holdout, tidak ada kontak dengan model selama training).

**Penghindaran Leakage**:
- Split berdasarkan **cow_id** (bukan per-gambar individual).
  - Alasan: 1 sapi mungkin memiliki 5–10 foto; jika split per-gambar, foto yang sama bisa tersebar di train & test.
  - Solusi: Kelompokkan semua foto 1 sapi → assign seluruhnya ke train/val/test.
- Jangan lakukan normalisasi global pada train/val/test gabungan.
  - Alasan: Statistik (mean, std) train akan bocor ke val/test.
  - Solusi: Hitung statistics hanya dari **train set** → terapkan ke val & test.

### 2.5 Validasi & Kontrol Kualitas Data

**Pemeriksaan pada Data Mentah (Raw)**:
- **Format file**: RGB `.jpg/.png` (8-bit), Depth `.png` (16-bit).
- **Dimensi**: Konsisten antar sampel (misal 1920×1440) — atau dokumentasikan jika bervariasi.
- **Rentang depth**: 0–8000 mm (typical untuk RealSense D435). Flag nilai anomali (misal 0 di ROI sapi).
- **File corrupt check**: Coba load setiap gambar; log error untuk diperbaiki/dihapus.

**Pemeriksaan pada Data Tersiap (Processed)**:
- **Kolom CSV**: Pastikan `cow_id`, `image_filename`, `bcs_label` ada dan valid.
- **Tipe data**: `bcs_label` numerik (float), nilai 1.00–5.00.
- **Missing values**: Tidak boleh ada NaN pada kolom kunci; log & dihapus jika ada.
- **Label distribution**: Visualisasi histogram BCS — pastikan tidak terlalu skewed (misal semua label 3.0).

**Pemeriksaan Sanity (Lapangan, Nanti)**:
- Bandingkan depth ekstrim (min/max) antar sapi → deteksi anomali.
- Visualisasi: RGB + overlay mask segmentasi (jika segmentation tersedia) → pastikan ROI masuk akal.

---

## 3) Desain Model (Simplified Pipeline)

### 3.1 Pendekatan Tahap Awal (PoC)

**Strategi Baseline** (sederhana, cepat):
- **Input**: Citra RGB saja (ukuran 224×224 atau 256×256).
- **Feature extraction**: Transfer learning dari pre-trained CNN (misal EfficientNet-B0, ResNet50).
- **Output layer**: Global average pooling → fully connected layers (128, 64 neurons) → 1 neuron (regresi BCS kontinu).
- **Alternative (optional)**: Ekstrak fitur secara manual dari histogram RGB, edge features, atau texture (untuk ablation study).

**Alasan Pilihan Ini**:
- **Transfer learning**: Model pre-trained pada ImageNet sudah belajar fitur visual generic (bentuk, tekstur). Hemat data & waktu training.
- **Sederhana & cepat**: Cocok untuk PoC, tidak perlu GPU powerful.
- **Mudah debug**: Jika performa rendah, bisa check aktivasi layer untuk diagnosis.

### 3.2 Model 1 — Feature Extraction (CNN Transfer Learning)

**Arsitektur**:
```
Input (RGB 224×224)
  ↓
[Backbone: EfficientNet-B0 pre-trained]
  ├─ Extract intermediate activations (misal layer sebelum global pool)
  ├─ Output: feature vector dimensi 1280 (atau sesuai model)
  ↓
[Global Average Pooling] → 1280-dim vector
  ↓
[Fully Connected]
  ├─ Dense(512) + ReLU
  ├─ Dropout(0.3)
  ├─ Dense(256) + ReLU
  ├─ Dropout(0.3)
  ├─ Dense(128) + ReLU
  ↓
[Output] → 1 neuron (linear activation, BCS prediction 1.0–5.0)
```

**Hyperparameter (Rekomendasi Awal)**:
- **Optimizer**: Adam (learning rate 1e-4 hingga 1e-3).
- **Loss function**: Mean Absolute Error (MAE) — lebih interpretable untuk regresi kontinu.
- **Batch size**: 16–32 (tergantung GPU/CPU).
- **Epochs**: 50–100 (dengan early stopping, patience=10).
- **Augmentasi data**:
  - Random horizontal flip (50%).
  - Random brightness/contrast adjustment (±10–20%).
  - Random rotation (±15°).
  - Random zoom (0.9–1.1×).
  - Gaussian blur ringan (σ=0.5–1.0).

**Training Strategy**:
- **Frozen backbone** (inisial): Setel backbone CNN *frozen* (tidak update weights), latih hanya top layers dahulu (~10 epoch).
- **Fine-tune backbone** (lanjutan): Unfreeze backbone, set learning rate lebih rendah (1e-5), lanjutkan training (~40 epoch).
- **Alasan**: Stabilitas training lebih baik, menghindari catastrophic forgetting dari pre-trained weights.

**Framework & Library**:
- **PyTorch** (recommended): `torch`, `torchvision`, `torch.nn`.
- **TensorFlow** (alternative): `tf.keras` dengan pre-trained models dari `tf.keras.applications`.

### 3.3 Model 2 — Regressor (Prediksi BCS)

Setelah training CNN di Section 3.2, **ekstrak feature vector** dari setiap gambar, lalu latih regressor ringan.

**Opsi Regressor**:

1. **MLPRegressor (Baseline)** — scikit-learn
   - Konfigurasi:
     ```python
     MLPRegressor(
         hidden_layer_sizes=(256, 128, 64),
         activation='relu',
         solver='adam',
         learning_rate='adaptive',
         max_iter=500,
         random_state=42,
         early_stopping=True,
         validation_fraction=0.1
     )
     ```
   - **Kelebihan**: Cepat, ringan, mudah di-save (joblib).
   - **Kekurangan**: Kurang robust terhadap outlier.

2. **XGBoost Regressor** — xgboost library (opsional)
   - Konfigurasi:
     ```python
     XGBRegressor(
         n_estimators=200,
         max_depth=5,
         learning_rate=0.01,
         objective='reg:squarederror',
         subsample=0.8,
         colsample_bytree=0.8,
         random_state=42,
         early_stopping_rounds=20
     )
     ```
   - **Kelebihan**: Robust, handal untuk outlier, sering menang kompetisi.
   - **Kekurangan**: Lebih kompleks, parameter tuning lebih banyak.

3. **Ordinal Regression** (opsional ablation)
   - Library: `mord` atau `coral-pytorch`.
   - Alasan: BCS adalah ordinal (1 < 2 < 3 < ... < 5), bukan nominal.
   - Performa: Kemungkinan lebih baik jika data terbatas.

**Rekomendasi Tahap Awal**: Mulai dengan **MLPRegressor**, kemudian coba **XGBoost** untuk comparison (ablation study).

### 3.4 Ekstraksi Fitur Manual (Alternatif Sederhana, Jika CNN Tidak Feasible)

Jika GPU tidak tersedia atau dataset terlalu kecil untuk CNN:

**Fitur Hand-Crafted** dari citra RGB:
- **Color histogram**: Jumlah bin per channel (R, G, B) → 30 fitur (10 bin × 3 channel).
- **Edge density**: Hitung edge filter (Sobel, Canny) → rasio piksel edge terhadap total.
- **Texture** (LBP / Haralick): Local Binary Pattern atau gray-level co-occurrence matrix.
- **Shape moments**: M00 (area), M10/M01 (centroid), M20 (variance), dll.
- **Total**: ~50–100 fitur, input untuk MLPRegressor atau XGBoost.

**Keuntungan**: Cepat, tidak perlu GPU, interpretable.
**Kekurangan**: Tidak sebaik deep learning, perlu domain knowledge untuk feature selection.

---

## 4) Evaluasi (Metodologi, Metrik, Interpretasi)

### 4.1 Protokol Evaluasi

**Cross-Validation (CV) pada Train Set**:
- **K-Fold CV**: 5-fold (K=5).
- **Stratifikasi**: Split berdasarkan **cow_id** (untuk data lapangan nanti) atau **BCS label bins** (untuk data publik, jika perlu balance).
- **Tujuan**: Estimate generalisasi model sebelum lihat test set; detect overfitting.
- **Output**: Rata-rata metrik ± std dev (misal MAE = 0.28 ± 0.04).

**Hold-Out Test Set**:
- **Komposisi**: 15% data publik yang tidak pernah disentuh selama training.
- **Evaluasi final**: Lapor metrik test set setelah semua training selesai.

**Field Validation (Nanti, Tahap Lanjut)**:
- Kumpulkan 10–20 sampel lapangan → rekayasa manual ground truth.
- Jalankan model → bandingkan prediksi vs. ground truth lapangan.
- Log failure cases, latensi, user feedback.

### 4.2 Metrik Utama

| Metrik | Formula / Deskripsi | Target (Tahap Awal) | Target (Akhir) | Interpretasi |
|--------|-------------------|-------------------|-----------------|-------------|
| **MAE** | (1/n) Σ \|y_pred - y_true\| | < 0.40 | < 0.25 | Error rata-rata (satuan BCS). Lebih interpretable. |
| **RMSE** | √[(1/n) Σ (y_pred - y_true)²] | < 0.50 | < 0.35 | Penalti lebih besar untuk error besar. |
| **R²** | 1 - (SS_res / SS_tot) | > 0.65 | > 0.75 | Varian target yang dijelaskan model (0–1). |
| **Pearson r** | Korelasi linear | > 0.75 | > 0.85 | Kekuatan hubungan linear prediksi vs. true. |
| **Acc@±0.25** | % prediksi dengan \|error\| ≤ 0.25 | ≥ 85% | ≥ 95% | Berapa persen prediksi "akurat" dalam ±0.25 BCS. |
| **Acc@±0.50** | % prediksi dengan \|error\| ≤ 0.50 | ≥ 95% | ≥ 98% | Toleransi lebih lebar untuk operasional. |

**Metrik Tambahan** (optional):
- **Bland–Altman plot**: Scatter (mean vs. difference) → deteksi bias sistematis.
- **Calibration plot**: Prediksi vs. true → visualisasi confidence.
- **Per-BCS MAE**: Breakdown metrik per kelas BCS (1.0, 1.5, 2.0, dll) → identifikasi kelas problematic.

### 4.3 Visualisasi & Diagnostik

**Plot Wajib** (dalam notebook evaluasi):
1. **Prediksi vs. True (scatter plot)**:
   - X-axis: true BCS, Y-axis: predicted BCS.
   - Diagonal y=x sebagai referensi (perfect prediction).
   - Warna/ukuran: Error magnitude.
   
2. **Residual plot**:
   - X-axis: predicted BCS, Y-axis: residual (pred - true).
   - Pastikan residual ~0 di semua prediksi (tidak ada bias).
   
3. **Error distribution (histogram)**:
   - X-axis: error magnitude, Y-axis: frequency.
   - Ekspektasi: distribusi simetris, mean ~0.
   
4. **Confusion matrix** (untuk classification-like analysis):
   - Jika label dibulatkan ke integer (1, 2, 3, 4, 5), buat confusion matrix kategorik.

5. **CV stability plot**:
   - Fold ke-i vs. metrik fold ke-i.
   - Pastikan CV scores konsisten (tidak ada fold yang sangat outlier).

### 4.4 Interpretasi Hasil & Troubleshooting

**Skenario: MAE tinggi (> 0.40) pada tahap awal**:
- ✓ **Normal**: Data publik mungkin dari populasi/kamera berbeda; masih dalam ekspektasi PoC.
- **Action**: Tunggu data lapangan untuk fine-tuning; atau coba augmentasi lebih agresif.

**Skenario: Overfitting (train MAE rendah, val/test MAE tinggi)**:
- **Penyebab**: Model terlalu kompleks, data terbatas, augmentasi kurang.
- **Solusi**:
  - Tambah regularisasi (L1/L2 di layer dense).
  - Tambah dropout rate.
  - Tambah augmentasi data.
  - Kurangi kompleksitas model (fewer neurons, shallower network).

**Skenario: Underfitting (train & val MAE sama-sama tinggi)**:
- **Penyebab**: Model terlalu sederhana, learning rate terlalu rendah, epochs kurang.
- **Solusi**:
  - Perbesar model (lebih layers/neurons).
  - Naikkan learning rate, training lebih lama.
  - Check data quality (label error? feature kuat?).

---

## 5) Implementasi: Training & Iterasi

### 5.1 Environment Setup

**Option 1: Google Colab (Gratis, GPU)**
- Pro: GPU Tesla T4 gratis, tidak perlu install lokal.
- Con: Session timeout, storage terbatas (15 GB).
- Langkah:
  1. Upload notebook `.ipynb` ke Google Drive.
  2. Buka via Google Colab.
  3. Mount drive: `from google.colab import drive; drive.mount('/content/drive')`.
  4. Install dependensi: `!pip install torch torchvision tensorflow pandas scikit-learn xgboost mlflow`.

**Option 2: Kaggle Notebook (Gratis, GPU/TPU)**
- Pro: Dataset bawaan, TPU tersedia, storage lebih besar.
- Con: Navigasi UI lebih rumit.
- Langkah: New notebook → add data → run cells.

**Option 3: Local / Cloud VM (Berbayar)**
- Pro: Kontrol penuh, GPU scalable.
- Con: Biaya, setup lebih kompleks.
- Tools: AWS SageMaker, Google AI Platform, Azure ML, atau VM biasa + NVIDIA docker.

### 5.2 Workflow Training (Pipeline dalam Satu Notebook)                                                                         
Pada fase ini, seluruh coding pipeline ML (mulai dari ringkasan business understanding singkat sampai evaluasi & penyimpanan/download model) dilakukan di **satu notebook utama**:
                          
`notebooks/01_bcs_full_pipeline.ipynb`. Kode **tidak dipecah** lagi menjadi banyak notebook/script; pemisahan hanya dilakukan untuk tahap deployment API (FastAPI) di folder `api`.

Di dalam notebook, tahapan dibagi per **segmen** (blok kode) sebagai berikut:

- **Segmen 1 – Setup Konteks**                                                            
  - Import library, set random seed, dan inisialisasi environment (opsional: baca `config.yaml`).                      
  - Cell markdown ringkasan business understanding, tujuan model, dan metrik utama (disarikan dari dokumen ini).                                                                                                                 
- **Segmen 2 – EDA & Data Understanding**
  - Load dataset publik (Dryad / UNICT).                                                                       
  - Visualisasi contoh citra + label. 
  - Statistik dasar: distribusi BCS, ukuran gambar, metadata penting.                                                   
  - Deteksi anomali / file corrupt sederhana.

- **Segmen 3 – Data Preparation & Splitting**
  - Normalisasi / resizing citra ke ukuran input backbone (mis. 224×224).                                                 
  - Augmentasi dasar (flip, brightness/contrast, rotation).                                                             
  - Split train/val/test berbasis `cow_id` untuk menghindari leakage.

- **Segmen 4 – Feature Extraction**   
  - Load backbone pre-trained (mis. EfficientNet-B0 via torchvision). 
  - Ekstraksi feature vector untuk train/val/test.

- **Segmen 5 – Training & Model Selection**
  - Training regressor (MLPRegressor / XGBoost) di atas fitur backbone.
  - 5-fold cross-validation, logging metrik (MAE, R², dsb.). 
  - Pemilihan konfigurasi/model terbaik berdasarkan metrik validasi.

- **Segmen 6 – Evaluation & Export Model** 
   - Evaluasi pada hold-out test set.
   - Visualisasi error (residual plot, distribusi error, dsb.).
   - Simpan artefak model (`models/cnn_backbone_v1.pt`, `models/bcs_regressor_v1.pkl`, metadata) sebagai output pipeline. 

Tahapan deployment API tidak ditulis di notebook, tetapi di modul Python terpisah (`api/main.py`, `api/inference.py`) yang hanya memuat artefak model hasil notebook dan menyediakan endpoint inferensi untuk dihubungkan ke backend/web dev. 

### 5.3 Konfigurasi & Hyperparameter

**File `config.yaml` (contoh)**:
```yaml
# Model
model:
  backbone: "efficientnet-b0"
  pretrained: true
  input_size: 224
  output_dim: 1

# Training
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  optimizer: "adam"
  loss: "mae"
  early_stopping_patience: 10
  weight_decay: 0.0001

# Data
data:
  train_fraction: 0.70
  val_fraction: 0.15
  test_fraction: 0.15
  augmentation:
    hflip: 0.5
    brightness: 0.15
    contrast: 0.15
    rotation: 15

# Evaluation
eval:
  cv_folds: 5
  metrics: ["mae", "rmse", "r2", "pearson_r"]

# MLflow
mlflow:
  enabled: true
  tracking_uri: "./mlruns"
  experiment_name: "bcs-v1-poc"
```

**Load config di notebook**:
```python
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

---

## 6) Deployment: API Inference

### 6.1 Arsitektur API

```
┌─────────────────┐
│  Mobile App     │  (atau web form)
│  - Capture RGB  │
│  - Compress     │
└────────┬────────┘
         │ POST /api/v1/predict (multipart/form-data)
         │
┌────────▼────────────────────────────────────────────┐
│  FastAPI Server (Python)                            │
│  - Validate input (format, size)                    │
│  - Load cached model (CNN + regressor)              │
│  - Preprocess: resize, normalize                    │
│  - Forward pass: extract feature → predict BCS      │
│  - Log request (timestamp, cow_id, latency)         │
└────────┬────────────────────────────────────────────┘
         │ JSON response { bcs_pred: 3.42, confidence: 0.87, ... }
         │
┌────────▼────────┐
│  Client         │  (parse & display)
│  - Show BCS     │
│  - Save history │
└─────────────────┘
```

### 6.2 Kontrak API (Spesifikasi)

**Endpoint**: `POST /api/v1/predict`

**Request (multipart/form-data)**:
```json
{
  "image": <file>,           // RGB JPG/PNG, max 10 MB
  "cow_id": "cow_001",       // optional
  "device_id": "iphone13",   // optional, untuk logging
  "timestamp": 1700000000    // optional, UNIX timestamp
}
```

**Response 200 OK**:
```json
{
  "status": "success",
  "bcs_pred": 3.42,
  "confidence": 0.87,
  "error_message": null,
  "latency_ms": 1240,
  "model_version": "v1-poc-20241117",
  "debug": {
    "input_shape": [224, 224, 3],
    "feature_vector_shape": [1280]
  }
}
```

**Response 400 Bad Request**:
```json
{
  "status": "error",
  "error_message": "Invalid image format. Expected JPG/PNG.",
  "bcs_pred": null
}
```

**Response 422 Unprocessable Entity** (citra tidak dapat diproses):
```json
{
  "status": "error",
  "error_message": "Unable to detect cow in image. Ensure image is clear and cow is visible.",
  "bcs_pred": null
}
```

### 6.3 Implementasi API Minimal (FastAPI)

**File `api/main.py` (skeleton)**:

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import joblib
import cv2
import numpy as np
import torch
import torchvision.models as models
import io
import time
from datetime import datetime

# Load model global (cache di memory)
MODEL_CNN = None
MODEL_REGRESSOR = None

app = FastAPI(title="BCS Inference API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_models():
    global MODEL_CNN, MODEL_REGRESSOR
    print("Loading models...")
    # Load CNN feature extractor
    MODEL_CNN = models.efficientnet_b0(pretrained=True)
    MODEL_CNN.classifier = torch.nn.Identity()  # Remove classification head
    MODEL_CNN.eval()
    
    # Load regressor
    MODEL_REGRESSOR = joblib.load("models/bcs_regressor_v1.pkl")
    print("Models loaded successfully.")

@app.post("/api/v1/predict")
async def predict(image: UploadFile = File(...), cow_id: str = None, device_id: str = None):
    """
    Predict BCS from cattle image.
    """
    start_time = time.time()
    
    try:
        # 1. Validate file
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image format. Expected JPG/PNG.")
        
        # 2. Read image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_rgb = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_rgb is None:
            raise HTTPException(status_code=422, detail="Unable to decode image.")
        
        # 3. Preprocess
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        img_tensor = (img_tensor - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_batch = img_tensor.unsqueeze(0)
        
        # 4. Feature extraction
        with torch.no_grad():
            features = MODEL_CNN(img_batch).numpy().flatten()
        
        # 5. Predict BCS
        bcs_pred = float(MODEL_REGRESSOR.predict([features])[0])
        bcs_pred = np.clip(bcs_pred, 1.0, 5.0)  # Clamp to valid range
        
        # 6. Calculate confidence (optional)
        confidence = 0.87  # Dummy; bisa diimplementasikan lebih sophisticated
        
        # 7. Latency
        latency_ms = (time.time() - start_time) * 1000
        
        return JSONResponse({
            "status": "success",
            "bcs_pred": round(bcs_pred, 2),
            "confidence": round(confidence, 2),
            "error_message": None,
            "latency_ms": round(latency_ms, 1),
            "model_version": "v1-poc-20241117",
            "debug": {
                "input_shape": [224, 224, 3],
                "feature_vector_shape": list(features.shape)
            }
        })
    
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "status": "error",
                "error_message": e.detail,
                "bcs_pred": None,
                "latency_ms": round((time.time() - start_time) * 1000, 1)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error_message": f"Internal server error: {str(e)}",
                "bcs_pred": None,
                "latency_ms": round((time.time() - start_time) * 1000, 1)
            }
        )

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Jalankan lokal**:
```bash
# Install dependencies
pip install fastapi uvicorn torch torchvision pillow python-multipart

# Run server
python api/main.py
```

**Test di terminal baru**:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "image=@/path/to/test_image.jpg" \
  -F "cow_id=cow_001"
```

### 6.4 Integrasi dengan Laravel Backend

**Alur komunikasi**:
1. Mobile/web form di Laravel frontend → POST `/submit_photo` ke Laravel backend.
2. Laravel backend → simpan file, extract metadata → POST ke Python API (`http://ml-server:8000/api/v1/predict`).
3. Python API → return BCS prediction.
4. Laravel backend → simpan hasil prediksi ke MySQL (`bcs_records` table).
5. Dashboard Laravel → tampilkan riwayat prediksi per sapi.

**Contoh Laravel controller** (pseudo-code):
```php
<?php
namespace App\Http\Controllers;

use Illuminate\Http\Request;
use App\Models\BcsRecord;
use App\Models\Cow;

class BcsController extends Controller
{
    public function predictBCS(Request $request)
    {
        // 1. Validate upload
        $request->validate(['image' => 'required|image|max:10000']);
        
        // 2. Save image temporarily
        $image = $request->file('image');
        $temp_path = $image->store('temp', 'local');
        
        // 3. Call Python ML API
        $client = new \GuzzleHttp\Client();
        $response = $client->post('http://ml-server:8000/api/v1/predict', [
            'multipart' => [
                ['name' => 'image', 'contents' => fopen(storage_path("app/$temp_path"), 'r')],
                ['name' => 'cow_id', 'contents' => $request->input('cow_id')],
            ]
        ]);
        
        $result = json_decode($response->getBody(), true);
        
        // 4. Save to database
        if ($result['status'] === 'success') {
            BcsRecord::create([
                'cow_id' => $request->input('cow_id'),
                'bcs_value' => $result['bcs_pred'],
                'image_path' => $temp_path,
                'model_version' => $result['model_version'],
                'latency_ms' => $result['latency_ms'],
                'predicted_at' => now(),
            ]);
            
            return response()->json(['success' => true, 'bcs' => $result['bcs_pred']]);
        } else {
            return response()->json(['error' => $result['error_message']], 422);
        }
    }
}
```

**Route** (`routes/api.php`):
```php
Route::post('/bcs/predict', [BcsController::class, 'predictBCS']);
```

---

## 7) MLOps & Tracking Eksperimen (Opsional)

### 7.1 MLflow Setup (untuk tracking)

**Instalasi**:
```bash
pip install mlflow
```

**Contoh logging** (dalam notebook training):
```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment("bcs-v1-poc")

with mlflow.start_run(run_name="baseline-mlp"):
    # Log parameters
    mlflow.log_param("model_type", "MLPRegressor")
    mlflow.log_param("hidden_layers", (256, 128, 64))
    mlflow.log_param("learning_rate", 0.001)
    
    # Log metrics
    for fold, (train_score, val_score) in enumerate(cv_results):
        mlflow.log_metric("train_mae", train_score, step=fold)
        mlflow.log_metric("val_mae", val_score, step=fold)
    
    # Log model
    mlflow.sklearn.log_model(regressor, "model")
    
    # Log artifacts
    mlflow.log_artifact("plots/residual_plot.png")
```

**View results**:
```bash
mlflow ui --backend-store-uri ./mlruns
# Open browser: http://localhost:5000
```

### 7.2 Version Control (Git + DVC)

**Git** (untuk code):
```bash
git init
git add .
git commit -m "Initial commit: BCS ML pipeline v1"
git branch -b feature/xgboost-comparison
```

**DVC** (untuk data/models, optional):
```bash
dvc init
dvc add data/processed/
dvc add models/bcs_regressor_v1.pkl
git add data/.gitignore, models/.gitignore, dvc.yaml
git commit -m "Track data & models with DVC"
```

---

## 8) QA, Testing & Field Validation

### 8.1 Unit Tests

**File `tests/test_inference.py`** (contoh):

```python
import pytest
import numpy as np
import cv2
from api.inference import extract_features, predict_bcs

def test_extract_features_shape():
    """Verifikasi output feature shape."""
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = extract_features(dummy_img)
    assert features.shape == (1280,), f"Expected (1280,), got {features.shape}"

def test_predict_bcs_range():
    """Prediksi BCS harus dalam range 1.0-5.0."""
    dummy_features = np.random.randn(1, 1280)
    bcs = predict_bcs(dummy_features)
    assert 1.0 <= bcs <= 5.0, f"BCS out of range: {bcs}"

def test_predict_bcs_deterministic():
    """Same input → same output."""
    features = np.random.randn(1, 1280)
    bcs1 = predict_bcs(features)
    bcs2 = predict_bcs(features)
    assert bcs1 == bcs2, "Model output tidak deterministic"
```

**Run tests**:
```bash
pytest tests/test_inference.py -v
```

### 8.2 Integration Tests

**File `tests/test_api.py`**:

```python
from fastapi.testclient import TestClient
from api.main import app
import io

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_valid_image():
    """Test dengan dummy image."""
    # Create 224x224 RGB dummy image
    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, img_bytes = cv2.imencode('.jpg', dummy_img)
    
    response = client.post(
        "/api/v1/predict",
        files={"image": ("test.jpg", io.BytesIO(img_bytes))},
        data={"cow_id": "test_001"}
    )
    assert response.status_code == 200
    result = response.json()
    assert "bcs_pred" in result
    assert 1.0 <= result["bcs_pred"] <= 5.0

def test_predict_invalid_format():
    """Test dengan format invalid."""
    response = client.post(
        "/api/v1/predict",
        files={"image": ("test.txt", io.BytesIO(b"invalid"))}
    )
    assert response.status_code == 400
```

### 8.3 Field Validation (Tahap Lanjut)

**Protokol minimal**:
- Kumpulkan 10–20 foto lapangan.
- Bandingkan prediksi model vs. ground truth manual.
- Log:
  - Prediksi BCS.
  - Ground truth (rata-rata 2 rater).
  - Confidence model.
  - Waktu inferensi.
  - Catatan operator (kondisi pencahayaan, posisi sapi, dll).

**Analisis**:
- MAE lapangan vs. train/val/test MAE publik → assess generalisasi.
- Failure cases: identifikasi kapan model salah → actionable insights.
- User feedback: usability, intuitif?, dokumentasi cukup?.

---

## 9) Roadmap & Estimasi Timeline

| Phase | Duration | Key Activities | Deliverable |
|-------|----------|----------------|------------|
| **1: Tech Specs & Desain** | Week 8–9 | Menyusun dan finalisasi dokumen `ml_tech_specs.md`, arsitektur sistem, dan desain pipeline (notebook ML + API) | `tech-specs/ml_tech_specs.md` siap implementasi |
| **2: Implementasi ML Pipeline (Notebook)** | Week 10–12 | Implementasi pipeline end-to-end di `notebooks/01_bcs_full_pipeline.ipynb` (EDA, data preparation, training, evaluasi, simpan model) | Notebook pipeline + artefak model di `models/` |
| **3: Implementasi API & Integrasi** | Week 13–14 | Implementasi FastAPI (`api/main.py`, `api/inference.py`), uji integrasi dengan backend/web dev, dan perbaikan minor hasil testing | Endpoint inferensi berfungsi dan terhubung ke backend |

---

## 10) Troubleshooting & FAQ

### Q: Bagaimana jika dataset publik tidak cocok dengan kondisi lapangan?

**A**: Transfer learning membantu, tapi fine-tuning perlu. Strategi:
1. Train di data publik (PoC).
2. Ambil 10–20 sampel lapangan → label manual.
3. Unfreeze model CNN → fine-tune di gabungan publik + lapangan (weighted balance).
4. Monitor: cek jangan overfit ke lapangan baru.

### Q: API melambat (latency > 5 detik)?

**A**: Kemungkinan:
- GPU tidak tersedia → coba CPU optimization (quantization, distillation).
- Model terlalu besar → gunakan lightweight backbone (MobileNet, SqueezeNet).
- I/O bottleneck → implement caching, batch processing.

### Q: Model prediksi BCS selalu 3.5 (mean)?

**A**: Model underfit atau fitur weak.
- Check label distribution (balance?).
- Visualisasi feature importance (if using tree models).
- Tambah augmentasi, normalisasi input.
- Coba model lebih kompleks (lebih neurons/layers).

### Q: Bagaimana reproducibility?

**A**: Simpan:
- `requirements.txt` (dependencies).
- `config.yaml` (hyperparameters).
- Model weights + version date.
- Random seed di setiap training.
- Metadata: train/val/test split indices (untuk replicability).

---

## 11) Referensi & Sumber Daya

### Data Publik
- **Dryad**: https://datadryad.org/ (cari "cattle BCS")
- **Penn Vet BCS Chart**: Visual reference untuk labeling manual.
- **Kaggle**: Cari dataset "cattle detection", "livestock", "bovine body condition".

### Library & Tools
- **Deep Learning**: PyTorch, TensorFlow/Keras.
- **ML**: scikit-learn, XGBoost.
- **API**: FastAPI, Flask.
- **Tracking**: MLflow, Weights & Biases.
- **Data**: Pandas, NumPy, OpenCV.

### Publikasi Referensi
- Literatur otomasi BCS berbasis citra (search: "automated body condition score cattle", "vision-based livestock monitoring").
- Multi-view 3D reconstruction untuk bovine morphology.

---

## 12) Lampiran

### 12.1 Checklist Implementasi Tahap Awal

- [ ] Data publik ter-download dan ter-validasi.
- [ ] Folder struktur sesuai `bcs-project/` template.
- [ ] Notebook pipeline (`01_bcs_full_pipeline.ipynb`) selesai; EDA & data split berdasarkan cow_id terdokumentasi.
- [ ] Model CNN + regressor berhasil train; CV MAE logged.
- [ ] Hold-out test evaluasi selesai; metrik dicatat.
- [ ] API skeleton berfungsi lokal (POST /api/v1/predict).
- [ ] Integration test berhasil (end-to-end citra → prediksi).
- [ ] Dokumentasi awal lengkap (README, API spec, troubleshooting).
- [ ] Repo git di-commit; models & data tracked (DVC optional).

### 12.2 Template requirements.txt (Python)

```txt
# Core ML
torch==2.2.0
torchvision==0.17.0
tensorflow==2.14.0
scikit-learn==1.4.0
numpy==1.26.0
pandas==2.2.0
opencv-python==4.8.1

# ML tools
xgboost==2.0.2
joblib==1.3.2
mlflow==2.12.0

# API & Web
fastapi==0.110.0
uvicorn[standard]==0.27.0
python-multipart==0.0.9
pydantic==2.5.0
requests==2.31.0

# Utilities
pyyaml==6.0
matplotlib==3.8.2
seaborn==0.13.1
jupyter==1.0.0

# Database (optional)
psycopg2-binary==2.9.9
sqlalchemy==2.0.23
```

### 12.3 Contoh Command Runbook (Makefile)

> Catatan: untuk fase PoC ini, pipeline utama (mulai dari ringkasan business understanding, EDA, data preparation, training, sampai evaluasi & penyimpanan model) dijalankan di **satu notebook**: `notebooks/01_bcs_full_pipeline.ipynb`. Tidak diperlukan script Python terpisah maupun dashboard; integrasi UI akan ditangani oleh tim web dev melalui backend/API.

```makefile
.PHONY: setup pipeline api test clean

setup:
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt

pipeline:
	jupyter notebook notebooks/01_bcs_full_pipeline.ipynb

api:
	python api/main.py

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache *.pyc
	find . -type d -name "__pycache__" -exec rm -r {} +
```

**Usage**:
```bash
make setup       # Setup environment
make pipeline    # Jalankan notebook pipeline end-to-end
make api         # Run API (deployment terpisah)
make test        # Unit tests
```

---

## 13) Pernyataan Kesiapan Produksi

**Status Dokumen**: Draft — Phase 1 (PoC)

**Next Steps untuk Phase 2 (Fine-Tuning)**:
1. Implementasi poin 1–5 di bagian Roadmap.
2. Kumpulkan 10–20 sampel lapangan.
3. Update dokumen ini dengan hasil field validation.
4. Finalisasi untuk deployment production.

**Kontak / Maintenance**:
- Tech Lead: [Name]
- ML Engineer: [Name]
- DevOps: [Name]

---

**Document Version**: 1.1 (2025-11-17)  
**Last Updated**: November 17, 2025  
**Status**: Ready for Phase 1 Implementation
