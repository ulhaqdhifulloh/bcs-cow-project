# BCS-COW-PROJECT (Modul ML)

Repositori ini berisi implementasi sistem Machine Learning untuk klasifikasi Body Condition Score (BCS) sapi menggunakan **YOLOv8-CLS**, sesuai spesifikasi teknis v1.3.

## Prasyarat

Pastikan Anda telah menginstal Python 3.11+ (rekomendasi spesifik python versi 3.11.9) dan pip.

## Struktur Direktori

- `api/`: Berisi kode untuk API (FastAPI) dan Frontend (Streamlit).
- `data/`: Folder penyimpanan dataset (raw dan processed).
- `models/`: Tempat penyimpanan model hasil training (`.pt`).
- `notebooks/`: Jupyter notebooks untuk eksperimen dan training pipeline.

## Cara Menjalankan

Ikuti langkah-langkah berikut secara berurutan:

### 1. Setup Environment

Masuk ke folder `ml/api` dan instal dependensi (disarankan menggunakan virtual environment):

```bash
cd api
pip install -r requirements.txt
```

### 2. Persiapan Data & Training

Gunakan notebook `notebooks/01_bcs_full_pipeline.ipynb` untuk:
1.  Memproses data raw menjadi format training.
2.  Melatih model YOLOv8.
3.  Mengevaluasi hasil training.

Model terbaik akan tersimpan di `ml/models/yolov8n_bcs_best.pt`.

### 3. Menjalankan API Server

Setelah model dilatih, Anda dapat menjalankan server API untuk inferensi.
Masuk ke folder `ml/api` dan jalankan:

```bash
uvicorn main:app --reload
```

Server akan berjalan di `http://localhost:8000`.

### 4. Menjalankan Aplikasi Web (Streamlit)

Untuk antarmuka pengguna yang lebih mudah, jalankan aplikasi Streamlit:

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser (biasanya `http://localhost:8501`).

### 5. Test API Manual

Anda juga dapat menguji API menggunakan curl atau Postman ke endpoint `/predict`:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/test_image.jpg"
```

Atau buka dokumentasi interaktif di `http://localhost:8000/docs`.

## 6. Viewing MLflow Experiment Tracking

Untuk melihat hasil logging dari eksperimen ML (metrics, parameters, artifacts), gunakan MLflow UI:

**PowerShell:**
```powershell
cd ml
mlflow ui --backend-store-uri "file:///$PWD/mlflow" --port 5000
```

**Command Prompt (CMD):**
```cmd
cd ml
mlflow ui --backend-store-uri "file:///%CD%/mlflow" --port 5000
```

Setelah server berjalan, buka browser di `http://127.0.0.1:5000` untuk melihat dashboard eksperimen.

> **Catatan**: Lihat [mlflow/README.md](mlflow/README.md) untuk dokumentasi lengkap tentang MLflow tracking.

