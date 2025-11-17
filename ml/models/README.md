# Models

Direktori ini menyimpan artefak model terlatih hasil pipeline di `notebooks/01_bcs_full_pipeline.ipynb`.

Contoh file yang dihasilkan (sesuai `ml_tech_specs_v1.1.md`):

- `cnn_backbone_v1.pt` – backbone CNN (transfer learning).
- `bcs_regressor_v1.pkl` – model regressor (MLPRegressor/XGBoost).
- `model_metadata.json` – metadata training (tanggal, metrik CV, dsb.).

Penamaan versi model dapat disesuaikan menggunakan suffix versi (mis. `_v1`, `_v2`) untuk tracking.

