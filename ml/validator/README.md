# Cow Rear Validator

Model binary classification untuk memvalidasi apakah gambar adalah bagian belakang sapi sebelum melakukan prediksi BCS.

## Tujuan

Mencegah prediksi BCS pada gambar yang bukan bagian belakang sapi (seperti bagian samping sapi, pintu, atau objek lain).

## Cara Penggunaan

### 1. Training Model

```bash
cd ml/validator
python cow_rear_validator_training.py
```

Script ini akan:
1. Mengambil 500 sample gambar dari dataset BCS sebagai kelas `cow_rear`
2. Download 500 gambar random dari internet sebagai kelas `not_cow_rear`
3. Melatih model YOLOv8n-cls selama 30 epochs
4. Menyimpan model ke `ml/models/cow_rear_validator/weights/best.pt`

### 2. Menggunakan Model

```python
from ultralytics import YOLO

# Load model
model = YOLO("../models/cow_rear_validator/weights/best.pt")

# Prediksi
results = model("path/to/image.jpg")
pred_class = results[0].probs.top1
pred_conf = results[0].probs.top1conf.item()
class_name = results[0].names[pred_class]  # 'cow_rear' atau 'not_cow_rear'

if class_name == 'not_cow_rear':
    print("Gambar bukan bagian belakang sapi!")
```

## Struktur Folder

```
validator/
├── README.md
├── cow_rear_validator_training.py
└── dataset/
    ├── data.yaml
    ├── train/
    │   ├── cow_rear/
    │   └── not_cow_rear/
    └── val/
        ├── cow_rear/
        └── not_cow_rear/
```

## Konfigurasi

Edit variabel di `cow_rear_validator_training.py` untuk menyesuaikan:

| Variabel | Default | Deskripsi |
|----------|---------|-----------|
| `NUM_COW_REAR_SAMPLES` | 500 | Jumlah sample dari dataset BCS |
| `NUM_NOT_COW_REAR_SAMPLES` | 500 | Jumlah gambar yang di-download |
| `EPOCHS` | 30 | Jumlah epoch training |
| `IMAGE_SIZE` | 224 | Ukuran gambar input |
