# Panduan Teknis Model Machine Learning: Sistem Klasifikasi Body Condition Score (BCS) Sapi
**Versi 1.3 — Evolusi: Dari PoC dengan Data Publik ke Implementasi dengan Data Raw & YOLOv8-CLS**

**Update**: Dokumen ini sekarang fokus **hanya** pada dataset dari Science Data Bank (Huang et al.) sebagai sumber data utama.

---

## Executive Summary

- **Fokus v1.3 (Updated)**: Implementasi praktis berbasis **YOLOv8-CLS** yang dioptimalkan untuk dataset Science Data Bank (53,566 images, 5 BCS classes). Sistem dirancang untuk **pipeline end-to-end single-fase** yang memanfaatkan satu sumber data authoritative.

- **Dataset**:
  - **Sumber Utama**: Dairy Cow Body Condition Score Target Detection Dataset (Huang Xiaoping et al., v3).
  - **Volume**: 53,566 images, 4.10 GB.
  - **Kelas**: 5 BCS scores (3.25, 3.50, 3.75, 4.00, 4.25).
  - **Lokasi Pengambilan**: Lu'an (Anhui), Huai'an (Jiangsu), Wuwei (Gansu).
  - **Validasi**: Labeling oleh tim veterinarian profesional.

- **Model Architecture**:
  - **Primary**: **YOLOv8-CLS (Classification)** — optimal untuk struktur dataset folder per kelas, inference cepat (<50ms), deployment simple.
  - **Secondary (Optional)**: CNN transfer learning (EfficientNet-B0) + Regressor — untuk validasi dan cross-check.

- **Target Performance**:
  - **Phase 1 (Current)**: MAE < 0.40; R² > 0.65; Top-1 accuracy ≥ 70% (YOLO).
  - **Phase 2+ (With Fine-tuning)**: MAE < 0.25; R² > 0.75; Top-1 accuracy ≥ 90% (YOLO); Accuracy@±0.25 ≥ 95%.

- **Inferensi**: API FastAPI ringan (<100ms latency) yang memuat model YOLOv8 (`.pt` atau `.onnx`), terintegrasi dengan backend Laravel untuk sistem manajemen peternakan.

- **Arsitektur Sistem**: Standalone ML inference → API server → Laravel dashboard.

---

## 1) Business Understanding & Strategi Implementasi

### 1.1 Latar Belakang & Masalah

**Tantangan Operasional**:
- BCS (Body Condition Score) penting untuk kesehatan reproduksi, produktivitas susu, dan manajemen bobot sapi.
- Penilaian manual subjektif (variabilitas antarrater ±0.5–1.0), padat karya, berpotensi menimbulkan stres hewan.
- Perlu solusi **akurat, objektif, non-invasif, praktis** untuk peternak kecil–menengah dengan konektivitas terbatas.

**Data Advantage** (Dataset Science Data Bank):
- **53,566 images** dari large-scale farming operations — sufisien untuk training robust model.
- **Professional validation** oleh veterinarian → high data quality & reliability.
- **Multi-region collection** (Anhui, Jiangsu, Gansu) → geographic diversity.
- **Structured in 5 discrete BCS classes** → perfectly aligned dengan classification task.

### 1.2 Tujuan & Cakupan

**Tujuan Utama**:
1. Mengotomasi estimasi BCS (5 discrete classes) dari citra RGB dengan akurasi ≥90%.
2. Membangun pipeline ML production-ready berbasis single authoritative dataset.
3. Menyediakan API inferensi + dashboard terintegrasi dengan Laravel.
4. Dokumentasi lengkap untuk reproducibility & handoff.

**Fase Implementasi** (Simplified: Single Phase Focused):

| Fase | Durasi | Dataset | Model | Target | Status |
|------|--------|---------|-------|--------|--------|
| **Phase 1** | 2–4 minggu | Science Data Bank (53,566 images) | YOLOv8-CLS | Top-1 ≥70% | **In Progress** |
| **Phase 2** | 1 minggu | Phase 1 model export | ONNX + Docker | Deployment ready | Pending Phase 1 |
| **Phase 3** | Ongoing | Production data | A/B testing, continuous improvement | MAE < 0.25 | Future |

**Cakupan v1.3**:
- ✅ Data pipeline: download, extract, split train/val/test.
- ✅ YOLOv8-CLS training dan evaluation.
- ✅ CNN transfer learning sebagai baseline validation (optional).
- ✅ Comprehensive Jupyter notebook.
- ✅ FastAPI inference server.
- ✅ Model export (`.pt`, `.onnx`).

**Di Luar Cakupan**:
- Multiple dataset sources (fokus single authoritative dataset saja).
- Depth camera atau sensor tambahan.
- Multi-spesies.
- On-device edge deployment (Phase 3+ consideration).

### 1.3 Metrik Sukses & Decision Criteria

**Teknis Phase 1 (PoC)**:

| Metrik | Target | Rationale |
|--------|--------|-----------|
| YOLOv8 Top-1 Accuracy | ≥ 70% | Baseline acceptable; adjacent classes mirip visually |
| YOLOv8 Top-2 Accuracy | ≥ 90% | Penting: Top-2 includes nearby BCS scores |
| CNN MAE (optional) | < 0.40 | Cross-validation; worst-case ±0.40 BCS |
| CNN R² (optional) | > 0.65 | Menjelaskan ≥65% variance |
| Training stability | No NaN/divergence | Convergence smooth |

**Teknis Phase 2+ (Production)**:

| Metrik | Target |
|--------|--------|
| YOLOv8 Top-1 Accuracy | ≥ 90% |
| Accuracy@±0.25 | ≥ 95% |
| API Latency | < 100 ms |
| Uptime (Production) | ≥ 99.0% |

---

## 2) Data (Akuisisi, Struktur, Pipeline)

### 2.1 Dataset Source & Karakteristik

**Dairy Cow Body Condition Score Target Detection Dataset**

| Atribut | Detail |
|---------|--------|
| **Source** | Science Data Bank (CSTR) |
| **DOI** | https://doi.org/10.57760/sciencedb.16704 |
| **CSTR Link** | https://cstr.cn/31253.11.sciencedb.16704 |
| **Version** | V3 (Latest: Feb 24, 2025) |
| **Total Images** | 53,566 |
| **Data Volume** | 4.10 GB |
| **File Format** | RAR archive (`dataset.rar`) |
| **MD5 Checksum** | `6a93c2714cfa4fb5875f7dae2017bb9b` |
| **BCS Classes** | 5 (3.25, 3.50, 3.75, 4.00, 4.25) |
| **Geographic Coverage** | Lu'an (Anhui), Huai'an (Jiangsu), Wuwei (Gansu) |
| **Labeling** | Professional veterinarian assessment |
| **License** | CC BY 4.0 (Attribution 4.0 International) |

**Authors**:
- Huang Xiaoping (Corresponding author: `hxping@mail.ustc.edu.cn`)
- Dou Zihao, Huang Fei, Zheng Huanyu, Hou Xiankun, Wang Chenyang, Feng Tao, Rao Yuan

**Funding**:
- Natural Science Foundation Project of Anhui Science and Technology Department (Grant No. 2308085MC103)
- Anhui Provincial Department of Education Natural Science Key Project (Grant No. KJ2021A0024)

### 2.2 Download & Ekstraksi Dataset

**Step 1: Download**
```bash
# Option A: Download via browser
# Link: https://cstr.cn/31253.11.sciencedb.16704.00CCAB0F.V2.V3.00F71B4C
# File: dataset.rar (4.10 GB)

# Option B: Download via command line (jika tersedia)
# curl atau wget ke CSTR link
```

**Step 2: Ekstraksi**
```bash
# Install unrar jika belum ada
# Ubuntu/Debian
sudo apt-get install unrar

# Extract RAR
unrar x dataset.rar -o+ ml/data/raw/

# Verify struktur
ls -la ml/data/raw/
# Expected output: folder 3.25/, 3.50/, 3.75/, 4.00/, 4.25/
```

**Step 3: Verify Data Integrity**
```bash
# Check MD5
md5sum dataset.rar
# Expected: 6a93c2714cfa4fb5875f7dae2017bb9b

# Count files per class
for class in 3.25 3.50 3.75 4.00 4.25; do
    count=$(find ml/data/raw/$class -type f | wc -l)
    echo "$class: $count images"
done
```

**Expected Structure**:
```
ml/data/raw/
├── 3.25/        # Images with BCS = 3.25
├── 3.50/        # Images with BCS = 3.50
├── 3.75/        # Images with BCS = 3.75
├── 4.00/        # Images with BCS = 4.00
└── 4.25/        # Images with BCS = 4.25
```

### 2.3 Data Pipeline: Raw → Processed

**Transformasi** (script `src/data/make_dataset.py`):

1. Baca semua gambar dari `ml/data/raw` (folder per kelas BCS).
2. Split stratified 70% train, 15% val, 15% test (berbasis image filename untuk consistency).
3. Copy ke struktur `ml/data/processed`:

```
ml/data/processed/
├── train/
│   ├── 3.25/
│   ├── 3.50/
│   ├── 3.75/
│   ├── 4.00/
│   └── 4.25/
├── val/
│   ├── 3.25/
│   ├── 3.50/
│   ├── 3.75/
│   ├── 4.00/
│   └── 4.25/
└── test/
    ├── 3.25/
    ├── 3.50/
    ├── 3.75/
    ├── 4.00/
    └── 4.25/
```

**Strategi Split** (berbasis image filename):
```python
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd

def make_dataset(raw_dir, processed_dir, test_size=0.15, val_size=0.15):
    """
    Transform raw folder structure (by BCS class) → train/val/test split.
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Create processed structure
    for split in ['train', 'val', 'test']:
        for bcs_class in ['3.25', '3.50', '3.75', '4.00', '4.25']:
            (processed_path / split / bcs_class).mkdir(parents=True, exist_ok=True)
    
    # Collect all files with metadata
    data_list = []
    for bcs_class in ['3.25', '3.50', '3.75', '4.00', '4.25']:
        class_dir = raw_path / bcs_class
        if class_dir.exists():
            for img_file in class_dir.glob('*.jpg'):
                data_list.append({'file': img_file, 'bcs': bcs_class})
    
    df = pd.DataFrame(data_list)
    
    # Stratified split by BCS class
    train_files, temp_files = train_test_split(
        df, test_size=(test_size + val_size), 
        random_state=42, stratify=df['bcs']
    )
    val_files, test_files = train_test_split(
        temp_files, test_size=test_size / (test_size + val_size),
        random_state=42, stratify=temp_files['bcs']
    )
    
    # Assign splits
    for df_split, split_name in [(train_files, 'train'), (val_files, 'val'), (test_files, 'test')]:
        for _, row in df_split.iterrows():
            src = row['file']
            dst_dir = processed_path / split_name / row['bcs']
            shutil.copy2(src, dst_dir / src.name)
    
    # Save metadata
    df_all = pd.concat([train_files, val_files, test_files])
    df_all.to_csv(processed_path / 'metadata.csv', index=False)
    
    print(f"Dataset processed:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val: {len(val_files)} images")
    print(f"  Test: {len(test_files)} images")
    print(f"  Total: {len(df_all)} images")

if __name__ == '__main__':
    make_dataset('ml/data/raw', 'ml/data/processed')
```

### 2.4 Augmentasi Data

**On-The-Fly Augmentation** (saat training):

- **Flip**: Horizontal flip (0.5 probability).
- **Color**: HSV augmentation ringan (H: ±5%, S: ±10%, V: ±10%).
- **Rotation**: ±10° (moderat).
- **Brightness/Contrast**: ±15%.
- **Scale**: Random crop 0.8–1.0×.

**Alasan**: Augmentasi meningkatkan generalisasi dan robustness terhadap variasi real-world (lighting, pose, angle).

### 2.5 Validasi Kualitas Data

**Checklist Raw Data**:
- ✓ Format file: `.jpg` atau `.png` (RGB 8-bit).
- ✓ Dimensi: Konsisten atau terdokumentasi.
- ✓ File corrupt check: Load setiap gambar, log error.
- ✓ Label match: Semua file di folder 3.25/ seharusnya BCS 3.25, dst.

**Checklist Processed Data**:
- ✓ CSV metadata: columns `file`, `bcs` ada dan valid.
- ✓ BCS label: Hanya values 3.25/3.50/3.75/4.00/4.25.
- ✓ Missing values: Tidak ada NaN.
- ✓ Label distribution: Histogram balanced per split.

---

## 3) Desain Model: YOLOv8-CLS (Primary)

### 3.1 Mengapa YOLOv8-CLS untuk Dataset Ini

**Kesesuaian Sempurna**:
- Dataset struktur folder per kelas → perfect untuk image classification task.
- Tidak perlu anotasi bounding box atau mask → langsung trainable.
- YOLOv8-CLS: ringan, cepat, pre-trained pada ImageNet.
- Output: discrete class probabilities (3.25, 3.50, 3.75, 4.00, 4.25).

**Performance Expectations**:
- Top-1 Accuracy: Prediksi exact match pada predicted class.
- Top-2 Accuracy: Prediksi benar dalam 2 class teratas (penting untuk BCS adjacent classes).
- Inference time: <50ms per gambar.

### 3.2 Arsitektur YOLOv8-CLS

```
Input RGB Image (224×224)
    ↓
[YOLOv8-Nano Backbone: CSPDarknet]
    ├─ Conv blocks + residuals
    ├─ Feature extraction
    ↓
[Global Average Pooling]
    ↓ 1280-dim feature vector
    ↓
[Classification Head]
    ├─ Linear layer
    ├─ Softmax (5 outputs)
    ↓
Output: [p_3.25, p_3.50, p_3.75, p_4.00, p_4.25]
```

### 3.3 Training Configuration

**File `config.yaml`**:
```yaml
model_yolo:
  pretrained: "yolov8n-cls.pt"      # nano untuk lightweight
  input_size: 224
  classes: 5                         # [3.25, 3.50, 3.75, 4.00, 4.25]
  model_name: "yolov8n-cls"

training:
  batch_size: 32                     # Adjust based on GPU memory
  num_epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  early_stopping_patience: 10
  yolo_device: 0                     # GPU device, 'cpu' untuk CPU

data:
  raw_dir: "ml/data/raw"
  processed_dir: "ml/data/processed"
  train_fraction: 0.70
  val_fraction: 0.15
  test_fraction: 0.15
  random_seed: 42
  
  augmentation:
    hflip: 0.5                       # Horizontal flip
    brightness: 0.15                 # ±15%
    contrast: 0.15                   # ±15%
    rotation: 10                     # ±10 degrees
    scale: [0.8, 1.0]               # Random crop

eval:
  cv_folds: 5
  metrics:
    - top1_accuracy
    - top2_accuracy
    - inference_time_ms
```

**Training Code**:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n-cls.pt')

# Train
results = model.train(
    data='ml/data/processed',        # train/val/test folders
    epochs=50,
    imgsz=224,
    batch=32,
    project='bcs_project',
    name='yolov8n_bcs_v1',
    patience=10,                      # Early stopping
    device=0,                         # GPU
    augment=True,
    val=True,
    save=True,
    verbose=True
)

# Evaluate
metrics = model.val()

# Export
model.export(format='onnx')
model.export(format='pt')
```

### 3.4 Baseline Validation: CNN Transfer Learning (Optional)

Untuk cross-validation, dapat training CNN (EfficientNet-B0) sebagai baseline comparison:

```python
import torch
import torch.nn as nn
from torchvision import models

class BCSRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.efficientnet_b0(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)  # Regression output
        )
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return torch.clamp(x, 1.0, 5.0)

# Training...
```

---

## 4) Training Pipeline (Phase 1)

### 4.1 Environment Setup

**Google Colab (RECOMMENDED)**:
```bash
from google.colab import drive
drive.mount('/content/drive')

!pip install -q ultralytics torch torchvision tensorflow scikit-learn xgboost pandas numpy matplotlib seaborn opencv-python
```

**Local/VM**:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4.2 Comprehensive Notebook: `01_bcs_full_pipeline.ipynb`

**Struktur 7 Segmen**:

1. **Setup & Context** (5 menit)
   - Import libraries, set seeds
   - Load config.yaml
   - Business objectives

2. **EDA & Data Understanding** (10 menit)
   - Load dataset dari `ml/data/raw`
   - Visualize 5 contoh per class
   - Statistik: distribution, size, metadata

3. **Data Preparation** (5 menit)
   - Run `make_dataset.py`
   - Verify train/val/test splits
   - Check leakage

4. **Train YOLOv8-CLS** (20 menit)
   - Load `yolov8n-cls.pt`
   - Configure & train
   - Save best model
   - Plot training curves

5. **Train CNN (Optional)** (15 menit)
   - Load EfficientNet-B0
   - Train regression head
   - Compare with YOLO

6. **Evaluation & Analysis** (15 menit)
   - YOLOv8: Top-1/Top-2 accuracy, confusion matrix, inference time
   - CNN: MAE, R² (if trained)
   - 5-fold CV results
   - Model selection

7. **Export & Serialize** (5 menit)
   - Save YOLO model (.pt, .onnx)
   - Save model_metadata.json
   - Summary report

---

## 5) Evaluasi Metrik

### 5.1 Metrik YOLOv8-CLS

| Metrik | Formula | Target (Phase 1) | Target (Phase 2+) | Unit |
|--------|---------|------------------|-------------------|------|
| **Top-1 Accuracy** | (# exact match) / total | ≥ 70% | ≥ 90% | % |
| **Top-2 Accuracy** | (# in top-2) / total | ≥ 90% | ≥ 98% | % |
| **Inference Time** | per image | < 100 ms | < 100 ms | ms |
| **Confusion Matrix** | Per-class breakdown | — | — | % |

### 5.2 Metrik CNN (Optional)

| Metrik | Formula | Target | Unit |
|--------|---------|--------|------|
| **MAE** | Σ\|pred - true\| / n | < 0.40 | BCS |
| **R²** | 1 - (SS_res / SS_tot) | > 0.65 | — |

### 5.3 Visualization

1. **Pred vs True (Scatter)** — ideal: points on y=x diagonal
2. **Confusion Matrix (YOLOv8)** — off-diagonal shows confusions
3. **Error Histogram** — distribution of prediction errors
4. **Training Curves** — loss & accuracy over epochs
5. **Inference Time Distribution** — latency histogram

---

## 6) Deployment: API Inferensi

### 6.1 FastAPI Server

**File `api/main.py`** (core implementation):

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None

app = FastAPI(
    title="BCS Inference API",
    version="1.0",
    description="Body Condition Score prediction for dairy cows"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_models():
    global MODEL
    try:
        logger.info("Loading YOLOv8-CLS model...")
        MODEL = YOLO('models/yolov8n_bcs_best.pt')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        MODEL.to(device)
        logger.info(f"Model loaded on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.post("/api/v1/predict")
async def predict(image: UploadFile = File(...), cow_id: str = None):
    start_time = time.time()
    
    try:
        if image.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=422, detail="Unable to decode image")
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = MODEL(img_rgb)
        result = results[0]
        
        top1_idx = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)
        
        bcs_classes = [3.25, 3.50, 3.75, 4.00, 4.25]
        bcs_pred = float(bcs_classes[top1_idx])
        
        latency_ms = (time.time() - start_time) * 1000
        
        return JSONResponse({
            "status": "success",
            "bcs_pred": round(bcs_pred, 2),
            "confidence": round(top1_conf, 3),
            "latency_ms": round(latency_ms, 1),
            "model_version": "v1-yolov8n-20251123",
            "debug": {
                "input_shape": list(img_rgb.shape),
                "model_type": "yolov8n-cls"
            }
        })
    
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"status": "error", "error_message": e.detail, "bcs_pred": None}
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error_message": str(e), "bcs_pred": None}
        )

@app.get("/health")
def health():
    return {"status": "healthy", "model": "yolov8n-cls"}

if __name__ == "__main__":
    import torch
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Run API**:
```bash
pip install fastapi uvicorn ultralytics torch opencv-python
python api/main.py
```

**Test**:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -F "image=@test_cow.jpg" \
  -F "cow_id=cow_001"
```

### 6.2 API Response Contract

**Request**:
```json
{
  "image": <file>,
  "cow_id": "cow_001"
}
```

**Response 200 OK**:
```json
{
  "status": "success",
  "bcs_pred": 3.5,
  "confidence": 0.92,
  "latency_ms": 45.2,
  "model_version": "v1-yolov8n-20251123",
  "debug": {
    "input_shape": [224, 224, 3],
    "model_type": "yolov8n-cls"
  }
}
```

### 6.3 Integrasi Laravel

**Controller** (`app/Http/Controllers/BCSController.php`):

```php
public function predictBCS(Request $request)
{
    $request->validate(['image' => 'required|image|max:10240']);
    
    $client = new \GuzzleHttp\Client();
    $response = $client->post(env('ML_API_URL') . '/api/v1/predict', [
        'multipart' => [
            ['name' => 'image', 'contents' => fopen($request->file('image')->path(), 'r')],
            ['name' => 'cow_id', 'contents' => $request->input('cow_id')]
        ]
    ]);
    
    $result = json_decode($response->getBody(), true);
    
    if ($result['status'] === 'success') {
        BcsRecord::create([
            'cow_id' => $request->input('cow_id'),
            'bcs_value' => $result['bcs_pred'],
            'confidence' => $result['confidence'],
            'model_version' => $result['model_version'],
            'predicted_at' => now()
        ]);
        
        return response()->json(['success' => true, 'bcs' => $result['bcs_pred']]);
    }
    
    return response()->json(['error' => $result['error_message']], 422);
}
```

---

## 7) Project Structure

```
bcs-cow-project/
├── README.md
├── requirements.txt
├── config.yaml
├── Makefile
├── .gitignore
│
├── data/
│   ├── raw/
│   │   ├── 3.25/
│   │   ├── 3.50/
│   │   ├── 3.75/
│   │   ├── 4.00/
│   │   └── 4.25/
│   │
│   └── processed/
│       ├── train/
│       ├── val/
│       ├── test/
│       └── metadata.csv
│
├── src/
│   ├── data/
│   │   ├── make_dataset.py
│   │   └── __init__.py
│   │
│   └── utils/
│       ├── metrics.py
│       ├── visualization.py
│       └── __init__.py
│
├── notebooks/
│   └── 01_bcs_full_pipeline.ipynb
│
├── models/
│   ├── yolov8n_bcs_best.pt
│   ├── yolov8n_bcs_best.onnx
│   └── model_metadata.json
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── config.py
│   └── requirements_api.txt
│
├── tests/
│   ├── test_inference.py
│   └── __init__.py
│
└── results/
    └── phase_1/
        ├── metrics.json
        └── plots/
```

---

## 8) Implementasi Multi-Fase (Simplified)

### Phase 1 (Saat Ini — 2–4 minggu): Development & Training

**Deliverables**:
- ✅ Data pipeline setup (download, extract, split).
- ✅ YOLOv8-CLS training & evaluation notebook.
- ✅ FastAPI inference server.
- ✅ Model artifacts (`.pt`, `.onnx`).
- ✅ Documentation v1.3.

**Acceptance Criteria**:
- [ ] YOLOv8-CLS Top-1 accuracy ≥ 70% pada test set.
- [ ] YOLOv8-CLS Top-2 accuracy ≥ 90%.
- [ ] API latency < 100 ms.
- [ ] Notebook reproducible di Google Colab.

---

### Phase 2 (1 minggu — Post-Phase 1): Production Deployment

**Prerequisites**: Phase 1 model approved.

**Deliverables**:
- ✅ Docker container image (FastAPI + model).
- ✅ Docker Compose setup.
- ✅ Kubernetes manifests (optional).
- ✅ Production runbook.
- ✅ Security & monitoring setup.

---

### Phase 3 (Ongoing): Continuous Improvement

**Deliverables**:
- ✅ Accuracy monitoring dashboard.
- ✅ A/B testing framework.
- ✅ Model retraining pipeline.

---

## 9) Quick Start

### 9.1 Prerequisites

**`requirements.txt`**:
```
ultralytics==8.0.200
torch==2.1.0
torchvision==0.16.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.2
opencv-python==4.8.1.78
matplotlib==3.8.2
seaborn==0.13.1
```

### 9.2 Setup (Google Colab)

```python
# Cell 1
!git clone https://github.com/your-repo/bcs-cow-project.git
%cd bcs-cow-project
!pip install -q -r requirements.txt

# Cell 2: Download dataset
# Manual download dari https://cstr.cn/31253.11.sciencedb.16704
# Upload dataset.rar ke Colab

# Cell 3: Extract & prepare
!mkdir -p ml/data/raw ml/data/processed
!cd ml/data && unrar x dataset.rar raw/

# Cell 4: Run pipeline
!python src/data/make_dataset.py
!jupyter notebook notebooks/01_bcs_full_pipeline.ipynb
```

### 9.3 Setup (Local)

```bash
git clone https://github.com/your-repo/bcs-cow-project.git
cd bcs-cow-project

# Create venv
python -m venv venv
source venv/bin/activate

# Install
pip install -r requirements.txt

# Download dataset (manual)
# Extract ke ml/data/raw/

# Prepare
python src/data/make_dataset.py

# Run notebook
jupyter notebook notebooks/01_bcs_full_pipeline.ipynb
```

---

## 10) References & Citation

### Dataset Citation

**CSTR Citation**:
```
Huang Xiaoping, Dou Zihao, Huang Fei, et al. 
Dairy cow body condition score target detection data set [DS/OL]. V3. 
Science Data Bank, 2025 [2025-11-20].
https://cstr.cn/31253.11.sciencedb.16704.
CSTR: 31253.11.sciencedb.16704.
```

**DOI Citation**:
```
Huang Xiaoping, Dou Zihao, Huang Fei, et al.
Dairy cow body condition score target detection data set [DS/OL]. V3.
Science Data Bank, 2025 [2025-11-20].
https://doi.org/10.57760/sciencedb.16704.
DOI: 10.57760/sciencedb.16704.
```

### Framework Documentation

- **YOLOv8**: https://docs.ultralytics.com/tasks/classify/
- **PyTorch**: https://pytorch.org/docs/
- **FastAPI**: https://fastapi.tiangolo.com/
- **scikit-learn**: https://scikit-learn.org/

---

## 11) FAQ & Troubleshooting

**Q: Dataset ukurannya 4.10 GB, apa bisa di-handle di Colab?**  
A: Ya, Colab punya ~50GB storage. Download langsung di Colab atau upload via Google Drive. Tips: ekstrak ke `/content/drive/MyDrive/` untuk persistent storage.

**Q: Berapa lama training YOLOv8-CLS dengan 53K images?**  
A: Batch=32, epochs=50, GPU Tesla T4 ~30–45 menit. CPU ~3–5 jam.

**Q: Bagaimana jika Top-1 accuracy < 70%?**  
A: (1) Check data quality (corrupt files). (2) Augmentasi lebih aggressive. (3) Longer training (epochs 100). (4) Larger model (yolov8s-cls). (5) Cross-check dengan CNN baseline.

**Q: Bisa deploy ke mobile?**  
A: Ya, Phase 2+ dengan ONNX export + TensorFlow Lite conversion.

**Q: Dataset ini CC BY 4.0, apa bisa komersial?**  
A: Ya, CC BY 4.0 memungkinkan penggunaan komersial dengan attribution. Pastikan cite paper & dataset.

---

## 12) Document History

| Version | Date | Changes |
|---------|------|---------|
| v1.3 | Nov 23, 2025 | **Updated: Fokus hanya dataset Science Data Bank (Huang et al.)**. Removal referensi dataset publik lain. Simplified roadmap (3 phases). YOLOv8-CLS sebagai primary model. |

---

**Dokumen ini adalah v1.3 FINAL (Updated) — Ready untuk Phase 1 Implementation.**

**Last Updated**: 23 November 2025, 11:29 WIB  
**Status**: ✅ Production-Ready / Phase 1 Ready  
**Dataset**: Science Data Bank - Dairy Cow BCS Detection Dataset (53,566 images)  
**Next Review**: Post-Phase 1 completion
