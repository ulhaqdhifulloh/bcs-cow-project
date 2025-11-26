# API & Aplikasi Klasifikasi Sapi BCS

Direktori ini berisi API dan aplikasi Streamlit untuk mengklasifikasikan Body Condition Score (BCS) sapi perah menggunakan model YOLOv8 yang telah dilatih.

## Struktur

- `main.py`: Aplikasi backend FastAPI.
- `streamlit_app.py`: Aplikasi frontend Streamlit.
- `Dockerfile`: Konfigurasi Docker untuk API.
- `requirements.txt`: Dependensi Python.

## Pengaturan (Setup)

### Prasyarat

- Python 3.11+ (rekomendasi spesifik python versi 3.11.9)
- Model YOLOv8 yang sudah dilatih berada di `../models/yolov8n_bcs_best.pt` (relatif terhadap direktori ini).

### Instalasi Lokal

1.  Buat virtual environment:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```
2.  Instal dependensi:
    ```bash
    pip install -r requirements.txt
    ```

## Menjalankan API

1.  Jalankan server FastAPI:
    ```bash
    uvicorn main:app --reload
    ```
2.  API akan tersedia di `http://localhost:8000`.
3.  Dokumentasi API: `http://localhost:8000/docs`.

## API Endpoints

Berikut adalah daftar endpoint yang tersedia:

-   **GET /**: Pembuka (Welcome message).
-   **GET /health**: Mengecek status API (Health check).
-   **POST /predict**: Melakukan prediksi (Upload image).

## Menjalankan Aplikasi Streamlit

1.  Di terminal baru, jalankan:
    ```bash
    streamlit run streamlit_app.py
    ```
2.  Aplikasi akan terbuka di browser Anda (biasanya di `http://localhost:8501`).

## Docker

Untuk membangun dan menjalankan API menggunakan Docker:

1.  Build image:
    ```bash
    docker build -t bcs-api .
    ```
2.  Jalankan container (dengan me-mount direktori models):
    ```bash
    # Asumsi Anda berada di ml/api dan models ada di ml/models
    # Anda perlu me-mount direktori models agar container dapat mengaksesnya.
    # Sesuaikan path ke direktori models Anda.
    docker run -p 8000:8000 -v "E:/Folder Kuliah/Semester 7/CAPSTONE PROJECT/Repository/bcs-cow-project/ml/models":/models -e MODEL_PATH=/models/yolov8n_bcs_best.pt bcs-api
    ```
