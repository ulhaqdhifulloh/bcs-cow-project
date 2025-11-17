# Panduan Teknis Model Machine Learning: Sistem Klasifikasi Body Condition Score (BCS) Sapi

Dokumen ini adalah spesifikasi teknis *end-to-end* untuk pengembangan, pelatihan, evaluasi, dan deployment model ML pada Proyek Capstone **“Pengembangan Sistem Klasifikasi Body Condition Score (BCS) pada Sapi Berbasis Depth Camera.”**

---

## Executive Summary
- **Data**: RGB + Depth 16-bit (RGB-D), masker segmentasi instance sapi.
- **Model**: Pipeline 2-tahap — (1) YOLOv8-Seg untuk segmentasi, (2) MLP Regressor untuk prediksi BCS dari fitur morfometrik (dari depth).
- **Target**: MAE < 0.25, R² > 0.75, *Accuracy@±0.25* ≥ 95%, latensi API < 5 detik/prediksi.
- **Arsitektur**: Hybrid Edge–Cloud (mobile akuisisi & offline cache; cloud untuk inferensi & dashboard).
- **Daya Uji**: 5-fold CV by **cow_id**, hold-out test set, uji lapangan di peternakan mitra.

---

## 1) Business Understanding

### 1.1 Latar Belakang & Masalah
- BCS penting untuk kesehatan reproduksi, produktivitas susu, dan manajemen bobot.
- Penilaian manual subjektif (variabilitas antarrater ±0.5–1.0), padat karya, dan berpotensi menimbulkan stres.
- Kebutuhan solusi **akurat, objektif, non-invasif**, dan **praktis** untuk peternak kecil–menengah; serta andal pada konektivitas terbatas.

### 1.2 Tujuan & Cakupan
- Mengotomasi estimasi BCS (skala 1–5, inkremen 0.25) dari citra RGB-D.
- Cakupan: akuisisi data, labeling, pelatihan model, API inferensi, dashboard, dan mode offline.
- Di luar cakupan awal: integrasi ke ERP peternakan, timbangan otomatis, multi-spesies.

### 1.3 Metrik Sukses
- **Teknis**: MAE < **0.25**; R² > **0.75**; korelasi r > **0.85**.
- **Akurasi operasional**: **Accuracy@±0.25 ≥ 95%** dan **Accuracy@±0.50 ≥ 98%**.
- **Sistem**: Latensi < **5 detik**/prediksi; *uptime* API ≥ **99.0%**; SUS > **70**.

---

## 2) Data (Akuisisi, Struktur, Kualitas)

### 2.1 Sumber Data
1. **Publik (baseline)**  
   - Dryad: *Labeled RGB and depth images for cattle BCS* (Intel RealSense D435i).  
   - UNICT + Penn Vet BCS DB (gambar top-view, anotasi anatomi & label BCS).
2. **Primer (lapangan) > Pending untuk Tahap Awal**  
   - Lokasi: Peternakan mitra (Jawa Barat).  
   - Target: **200–300** set RGB-D dari beberapa ekor sapi yang ada di peternakan tersebut (Holstein/cross).  
   - Perangkat: Kamera handphone/action cam/deep cam; tripod; penggaris/marker jarak.

### 2.2 Protokol Akuisisi
- **Jarak kamera**: 2–3 m dari sapi, tinggi ~1.2–1.6 m (disesuaikan).
- **Posisi**: standing; sudut *lateral* dan *posterior-anterior* (opsional *top-view* bila memungkinkan).
- **Variasi**: 20-30 *capture* per sapi; kondisi pencahayaan beragam (pagi/siang/sore).
- **Sinkronisasi label**: *Expert scoring* dilakukan pada jendela ≤ 1 jam dari pengambilan gambar.
- **Keamanan**: jarak aman, minim gangguan/stres.

### 2.3 Labeling (*Ground Truth*)
- Skala **1.00–5.00** dengan inkremen **0.25** (mis. 2.75).  
- **Dua rater** independen (dokter hewan/ahli). Simpan keduanya; gunakan **rata-rata** sebagai label target; log **perbedaan** untuk analisis reliabilitas.  
- Ukur **inter-rater reliability**: *weighted Cohen’s κ* dan/atau **ICC(2,k)**.  
- Simpan metadata: `cow_id`, umur, laktasi, waktu pengambilan, lokasi, perangkat, operator.

### 2.4 Struktur Folder & Skema Berkas
```text
bcs-project/
├── data/
│   ├── train/
│   │   ├── rgb/            # *.jpg / *.png (8-bit)
│   │   ├── depth/          # *.png (16-bit, raw depth in mm)
│   │   ├── masks/          # *.png (0/255) hasil Model 1
│   │   └── labels.csv      # cow_id,image_rgb,image_depth,bcs_label
│   ├── val/
│   │   ├── rgb/
│   │   ├── depth/
│   │   ├── masks/
│   │   └── labels.csv
│   └── test/
│       ├── rgb/
│       ├── depth/
│       ├── masks/
│       └── labels.csv
├── data_seg/               # Dataset YOLO (mask segmentation, format YOLO)
├── models/
│   ├── seg_yolov8s.pt      # Model 1
│   └── bcs_regressor.pkl   # Model 2
├── api/                    # FastAPI + inference pipeline
├── notebooks/              # EDA, prototyping
└── mlruns/                 # MLflow tracking (opsional)
```

### 2.5 *Data Split* & Kebocoran
- Splitting **berdasarkan `cow_id`** (bukan per-gambar) untuk menghindari *leakage* antar split.  
- Rekomendasi: **70/15/15** (train/val/test) atau **80/10/10** bila data terbatas.

### 2.6 Validasi & Kualitas Data
- **Depth** wajib 16-bit; verifikasi rentang (mm).  
- Deteksi file korup/hilang; *schema check* kolom CSV; ukuran gambar konsisten.  
- *Sanity check*: periksa konsistensi mask terhadap objek pada RGB/Depth.

---

## 3) Desain Model (Pipeline Dua-Tahap)

### 3.1 Model 1 — Segmentasi (YOLOv8-Seg)
- **Tujuan**: isolasi tubuh sapi dari latar (*overlapping* ternak & latar kompleks).  
- **Model**: `yolov8s-seg.pt` (ultralytics).  
- **Augmentasi**: *hflip*, brightness/contrast, blur ringan, *random shadow* (untuk robust terhadap pencahayaan).  
- **Latihan** (contoh skrip):
```python
from ultralytics import YOLO

seg_model = YOLO("yolov8s-seg.pt")  # pre-trained
seg_model.train(
    data="yolo_seg_data.yaml",  # path dataset YOLO (train/val, names: ["cow"])
    epochs=50,
    imgsz=640,
    batch=8,
    patience=10,
    optimizer="Adam",
    lr0=0.001,
)
# Model terbaik otomatis tersimpan ke runs/…/weights/best.pt -> pindahkan ke models/seg_yolov8s.pt
```

### 3.2 Ekstraksi Fitur Morfometrik (dari Depth + Mask)
- **ROI**: depth dibatasi masker (0/1).  
- **Fitur minimal (5)**:
  - `mean_depth`
  - `depth_variance`
  - `area_px` (jumlah piksel ROI)
  - `hw_ratio` (rasio tinggi-lebar bounding box ROI)
  - `body_volume_est` (∑ nilai depth pada ROI, proksi volume)
- Simpan ke `features.csv` (train/val/test).

### 3.3 Model 2 — Regressor (Prediksi BCS)
- **Baseline**: `MLPRegressor` (scikit-learn).  
- **Alternatif (opsional)**:
  - Ordinal regression (`mord.LogisticAT`) — karena label BCS bersifat ordinal.
  - XGBoost Regressor (nonlinear, kuat terhadap outlier).
  - *End-to-end* CNN regressor (EfficientNet-B0) sebagai ablation study.
- **Pelatihan** (contoh skrip, baseline):
```python
import pandas as pd
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score

train_fe = pd.read_csv("data/train/features.csv")
FEATURES = ["mean_depth","depth_variance","area_px","hw_ratio","body_volume_est"]
X, y = train_fe[FEATURES].values, train_fe["bcs_label"].values

reg = MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                   random_state=42, max_iter=600)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_mae = -cross_val_score(reg, X, y, cv=kf, scoring="neg_mean_absolute_error")
print(f"5-fold MAE: {cv_mae.mean():.3f} ± {cv_mae.std():.3f}")

reg.fit(X, y)
joblib.dump(reg, "models/bcs_regressor.pkl")
```

---

## 4) Evaluasi (Metodologi & Metrik)

### 4.1 Protokol Evaluasi
- **Cross-validation**: 5-fold by `cow_id` pada data train.  
- **Hold-out test**: evaluasi final pada `data/test`.  
- **Uji lapangan**: minimal 2 minggu, 2–3 lokasi; logging latensi, *failure rate*, dan feedback pengguna.

### 4.2 Metrik Utama
- **MAE** (target < 0.25).  
- **R²** (target > 0.75).  
- **Accuracy@±0.25**: % prediksi dengan |error| ≤ 0.25 (target ≥ 95%).  
- **Accuracy@±0.50**: % prediksi dengan |error| ≤ 0.50 (target ≥ 98%).  
- **Reliabilitas labeling**: κ tertimbang / ICC.  
- **Usability**: SUS > 70.

### 4.3 Analisis Tambahan
- Kurva *error vs. true BCS* (bias rendah/tinggi).  
- *Bland–Altman* vs. rata-rata *expert*.  
- Bootstrapping 1000× untuk CI 95% pada MAE dan Accuracy@±0.25.

---

## 5) Deployment (Arsitektur & API)

### 5.1 Arsitektur Hybrid Edge–Cloud
- **Edge (Mobile/App)**: akuisisi RGB-D, *pre-check* kualitas, kompresi, **offline cache** (SQLite), sinkronisasi saat online.  
- **Cloud**: FastAPI + model server, penyimpanan (S3/GCS), PostgreSQL, dashboard (Streamlit/Next.js), autentikasi JWT/OAuth2.

### 5.2 Kontrak API (contoh)
**Endpoint**: `POST /api/v1/predict`  
**Body (multipart/form-data)**:
- `image_rgb`: file `.jpg/.png` (RGB 8-bit)  
- `image_depth`: file `.png` (Depth 16-bit)  
- `cow_id` (opsional), `device_id` (opsional), `ts`

**Respons (200)**
```json
{
  "bcs_pred": 3.42,
  "agreement_±0.25": true,
  "latency_ms": 1320,
  "model_version": "2025.11.0",
  "debug": {"seg_det": 1, "roi_area_px": 42173}
}
```

### 5.3 Pipeline Inferensi (Server)
1. Validasi berkas & tipe.  
2. Load (cache) `seg_yolov8s.pt` & `bcs_regressor.pkl`.  
3. Inferensi segmentasi pada RGB → gabung semua instance menjadi satu mask biner.  
4. Baca depth 16-bit → normalisasi/cek rentang → terapkan mask (ROI).  
5. Hitung 5 fitur → `reg.predict()` → `bcs_pred`.  
6. Kembalikan JSON + metrik *debug* (opsional).

### 5.4 Non-fungsional
- **Keamanan**: JWT, batas ukuran unggah, skan malware, *rate limit*.  
- **Logging**: request idempotent, latensi, *error*, *drift signals*.  
- **Observabilitas**: Prometheus + Grafana (latensi, throughput, GPU/CPU).  
- **Skalabilitas**: autoscaling HPA; *queue* (RQ/Celery) untuk *batch*.  
- **Fallback**: Jika mask gagal, kirim *error code* yang terstandardisasi.

---

## 6) Reproducibility, MLOps, & Kode

### 6.1 Dependensi (contoh `requirements.txt`)
```txt
ultralytics==8.3.0
torch>=2.2.0
torchvision>=0.17.0
opencv-python>=4.8.1
scikit-learn>=1.4.0
joblib>=1.3.2
numpy>=1.26.0
pandas>=2.2.0
fastapi>=0.110.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.9
pydantic>=2.5.0
mlflow>=2.12.0     # opsional
psycopg2-binary>=2.9.9  # jika pakai PostgreSQL
```

### 6.2 Versi & Pelacakan Eksperimen
- Gunakan **MLflow** (atau Weights & Biases) untuk melacak eksperimen, metrik, artefak.  
- Gunakan **DVC** (opsional) untuk *versioning* data/mask/fitur.

### 6.3 Struktur Repo & Makefile (contoh)
```Makefile
.PHONY: setup train-seg infer-mask fe train-reg eval api

setup:
\tpip install -r requirements.txt

train-seg:
\tpython scripts/train_seg.py

infer-mask:
\tpython scripts/infer_seg_to_masks.py --split train --split val

fe:
\tpython scripts/extract_features.py

train-reg:
\tpython scripts/train_regressor.py

eval:
\tpython scripts/evaluate.py --split test

api:
\tuvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 7) Pengujian, QA, & Uji Lapangan
- **Unit test**: loader depth 16-bit, masker ROI, ekstraksi fitur, API kontrak.  
- **Integration test**: *end-to-end* batch kecil (10 sampel) dari unggah → respons.  
- **Field test**: log kegagalan (mask=0, depth out-of-range), waktu proses, *user notes*.

---

## 8) Risiko & Mitigasi
- **Variasi pencahayaan/lingkungan** → augmentasi kuat, kalibrasi kamera, *quality gate*.  
- **Occlusion/overlap** → YOLOv8-Seg instance mask; *min area threshold*.  
- **Drift populasi (peternakan baru)** → pemantauan *data drift*, *active learning* untuk labeling ulang.  
- **Data terbatas** → *transfer learning*, *mixup/cutout*, *hyperparameter search* ringan.  
- **Konektivitas buruk** → mode offline + *retry & backoff*.

---

## 9) Roadmap & Estimasi
- **Minggu 1–2**: setup data & labeling; baseline seg + fitur.  
- **Minggu 3–4**: pelatihan regressor; CV; ablation.  
- **Minggu 5**: integrasi API + dashboard; uji internal.  
- **Minggu 6–7**: uji lapangan; perbaikan; *hardening*.  
- **Minggu 8**: rilis pilot v1.0.

---

## 10) Lampiran

### 10.1 Contoh `yolo_seg_data.yaml`
```yaml
path: data_seg
train: train/images
val: val/images
names: ["cow"]
```

### 10.2 Contoh Endpoint FastAPI Minimal
```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import joblib, cv2, numpy as np

app = FastAPI()
# cache global
SEG_MODEL = None
REG = None

@app.on_event("startup")
def load_models():
    global SEG_MODEL, REG
    from ultralytics import YOLO
    SEG_MODEL = YOLO("models/seg_yolov8s.pt")
    REG = joblib.load("models/bcs_regressor.pkl")

@app.post("/api/v1/predict")
async def predict(image_rgb: UploadFile = File(...), image_depth: UploadFile = File(...)):
    # 1) load bytes -> ndarray (disederhanakan)
    rgb = np.frombuffer(await image_rgb.read(), dtype=np.uint8)
    rgb = cv2.imdecode(rgb, cv2.IMREAD_COLOR)
    depth_bytes = np.frombuffer(await image_depth.read(), dtype=np.uint8)
    depth = cv2.imdecode(depth_bytes, cv2.IMREAD_UNCHANGED)  # 16-bit

    # 2) seg
    seg = SEG_MODEL.predict(rgb, verbose=False)[0]
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    for m in getattr(seg, "masks", []) or []:
        mask = cv2.bitwise_or(mask, (m.data[0].cpu().numpy()*255).astype(np.uint8))

    if mask.sum() == 0:
        return JSONResponse({"error": "no_cow_detected"}, status_code=422)

    # 3) ROI depth
    if depth.dtype != np.uint16:
        return JSONResponse({"error": "invalid_depth_format"}, status_code=422)
    roi = depth.copy()
    roi[mask == 0] = 0

    # 4) fitur
    ys, xs = np.where(mask > 0)
    h, w = (ys.max()-ys.min()+1, xs.max()-xs.min()+1)
    mean_depth = float(roi[mask>0].mean())
    depth_var  = float(roi[mask>0].var())
    area_px    = int((mask>0).sum())
    hw_ratio   = float(h / max(w,1))
    body_vol   = float(roi.sum())

    feats = np.array([[mean_depth, depth_var, area_px, hw_ratio, body_vol]], dtype=np.float32)

    # 5) pred
    bcs_pred = float(REG.predict(feats)[0])
    return {"bcs_pred": round(bcs_pred, 2)}
```

### 10.3 Lisensi & Kepatuhan
- Pastikan kepatuhan lisensi dataset publik (Dryad/UNICT).  
- Lindungi data lapangan (PII pemilik, lokasi) sesuai kebijakan internal.

---

## 11) Referensi (non-ekshaustif)
- Dataset Dryad: *Labeled RGB and depth images for cattle BCS* (RealSense D435i).  
- UNICT + Penn Vet BCS Database.  
- Literatur otomasi BCS berbasis visi komputer (CNN/YOLO/3D point cloud).


---