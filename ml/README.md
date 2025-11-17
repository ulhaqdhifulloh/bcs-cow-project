# BCS-COW-PROJECT

Repositori ini berisi implementasi sistem Machine Learning untuk klasifikasi Body Condition Score (BCS) sapi, sesuai spesifikasi teknis di `tech-specs/ml_tech_specs_v1.1.md`.

Struktur folder utama mengikuti template `bcs-project/` pada dokumen tersebut, dengan penyesuaian nama root menjadi direktori proyek ini:

- `data/` – data publik & internal (raw, processed, field_data).
- `notebooks/` – notebook Jupyter utama pipeline ML.
- `models/` – artefak model terlatih.
- `api/` – service FastAPI untuk deployment model.
- `mlflow/` – tracking eksperimen (opsional).
- `tests/` – unit & integration tests.
- `web/` – kode frontend/backend web dari tim web dev.
- `tech-specs/` – dokumen spesifikasi teknis.

Detail lengkap alur ML pipeline, format data, dan kontrak API dapat dilihat di `tech-specs/ml_tech_specs_v1.1.md`.

