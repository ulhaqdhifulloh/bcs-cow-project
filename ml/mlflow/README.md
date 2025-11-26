# MLflow Experiment Tracking

Direktori ini berisi hasil tracking dari eksperimen Machine Learning menggunakan **MLflow**.

## Struktur Direktori

```
mlflow/
├── <experiment_id>/          # Folder per eksperimen (e.g., 999992508245777105)
│   ├── meta.yaml             # Metadata eksperimen
│   └── <run_id>/             # Folder per run (e.g., 1499a5d1f24546c38a86d4bcf3e5b688)
│       ├── meta.yaml         # Metadata run
│       ├── artifacts/        # Model artifacts, plots, files
│       ├── metrics/          # Logged metrics (accuracy, loss, dll)
│       ├── params/           # Logged parameters (hyperparameters)
│       └── tags/             # Run tags
└── .trash/                   # Archived/deleted runs
```

## Eksperimen Aktif

### BCS-Cattle-Classification
- **Experiment ID**: `999992508245777105`
- **Deskripsi**: Klasifikasi Body Condition Score (BCS) sapi menggunakan YOLOv8-CLS
- **Active Run ID**: `1499a5d1f24546c38a86d4bcf3e5b688`
- **Model**: YOLOv8-Nano Classification
- **Performance**: 89.84% Top-1 Accuracy, 32.99ms inference time

## Cara Menggunakan MLflow UI

MLflow UI memungkinkan Anda melihat dan membandingkan hasil training secara visual.

### Menjalankan MLflow UI

**PowerShell:**
```powershell
mlflow ui --backend-store-uri "file:///$PWD/mlflow" --port 5000
```

**Command Prompt (CMD):**
```cmd
mlflow ui --backend-store-uri "file:///%CD%/mlflow" --port 5000
```

Akses dashboard di: **http://127.0.0.1:5000**

### Fitur MLflow UI

1. **Experiments View**: Daftar semua eksperimen dan runs
2. **Metrics Comparison**: Bandingkan metrics antar runs
3. **Parameters**: Lihat hyperparameters yang digunakan
4. **Artifacts**: Download model, plots, dan file lainnya
5. **Search & Filter**: Cari runs berdasarkan metrics/parameters

## Logged Data

Setiap run mencatat:

### Metrics
- `accuracy` - Top-1 test accuracy
- `inference_time_ms` - Rata-rata waktu inferensi per gambar
- `train_loss` - Training loss (per epoch)
- `val_loss` - Validation loss (per epoch)

### Parameters
- `batch_size` - Ukuran batch training
- `epochs` - Jumlah epoch
- `learning_rate` - Learning rate
- `optimizer` - Optimizer yang digunakan
- `model_name` - Nama model (YOLOv8-cls)
- `input_size` - Resolusi input (224x224)

### Artifacts
- `production_bundle/` - Model production-ready
  - `best.pt` - PyTorch model
  - `best.onnx` - ONNX export
  - `metadata.json` - Model metadata
  - `README.md` - Release notes
- `training_curves.png` - Plot loss dan accuracy
- `confusion_matrix.png` - Confusion matrix test set

## Tips Penggunaan

### 1. Membandingkan Runs
- Centang beberapa runs di UI
- Klik "Compare" untuk melihat perbedaan metrics/parameters

### 2. Download Model Best
- Masuk ke run yang diinginkan
- Klik tab "Artifacts"
- Download `production_bundle/best.pt`

### 3. Menghapus Old Experiments
- Eksperimen dengan prefix `ARCHIVED_` bisa dihapus untuk menghemat ruang
- Centang eksperimen → Klik delete

### 4. Export Data
- Gunakan MLflow SDK untuk export data programmatically:
  ```python
  import mlflow
  
  mlflow.set_tracking_uri("file:///path/to/mlflow")
  run = mlflow.get_run("1499a5d1f24546c38a86d4bcf3e5b688")
  print(run.data.metrics)
  print(run.data.params)
  ```

## Troubleshooting

### Error: `Yaml file does not exist`
**Penyebab**: Ada folder non-MLflow di direktori mlflow.  
**Solusi**: Pindahkan folder yang bukan eksperimen ke `archive/`.

### Port already in use
**Penyebab**: MLflow UI masih berjalan di background.  
**Solusi**: Matikan proses lama atau gunakan port lain (e.g., `--port 5001`).

### Data tidak muncul di UI
**Penyebab**: `--backend-store-uri` tidak mengarah ke folder yang benar.  
**Solusi**: Pastikan menggunakan perintah dengan `file:///$PWD/mlflow` atau full path absolut.

## Referensi

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Tech Specs v1.4](../tech-specs/ml_tech_specs_v1.4.md)
