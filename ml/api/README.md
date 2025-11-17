# API Service

Direktori ini berisi service FastAPI untuk deployment model BCS.

Struktur awal:

- `main.py` – entrypoint FastAPI (routing & wiring).
- `inference.py` – fungsi untuk load model dan melakukan prediksi.
- `schemas.py` – Pydantic models untuk request/response.
- `database.py` – placeholder koneksi database (opsional).
- `config.py` – path model dan konfigurasi sederhana.
- `requirements_api.txt` – dependensi khusus service API.

Implementasi detil disesuaikan dengan kebutuhan integrasi ke backend Laravel/web dev.

