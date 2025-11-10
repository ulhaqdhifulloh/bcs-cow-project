# 🧠 TECH PROJECT TEMPLATE (Aplikasi / Website / AI / Automation)

> Tinggal duplikasi & isi. Hapus bagian yang tidak relevan.

---

## 1) Overview
- **Nama Proyek:** Sistem Monitoring Body Condition Score (BCS) Sapi Otomatis
- **Jenis:** Sistem AI/ML (Computer Vision) dengan platform Web/Mobile
- **Tujuan Utama:** Mengembangkan sistem monitoring BCS sapi yang otomatis, objektif, dan non-invasif menggunakan depth camera dan deep learning untuk meningkatkan efisiensi manajemen peternakan.
- **Status:** Prototipe (Pengembangan Capstone Design)

---

## 2) Problem & Solution
- **Masalah Pengguna:** Penilaian BCS manual bersifat subjektif (variabilitas ±0,5-1,0 poin ), memakan waktu (hanya 36% peternak monitoring rutin ), memerlukan keahlian khusus, dan membuat ternak stres. Adopsi teknologi terhambat literasi digital rendah (73,33% lulusan SD di Lembang) dan konektivitas internet terbatas.
- **Solusi yang Diajukan:** Sistem hybrid edge-cloud
  1. Akuisisi Data: Menggunakan depth camera (Intel RealSense/Azure Kinect) untuk menangkap data RGB-D (gambar dan kedalaman).
  2. Segmentasi: Model YOLOv8-Seg memisahkan tubuh sapi dari latar belakang, bahkan saat overlapping.
  3. Ekstraksi Fitur: Sistem mengekstraksi fitur morfometrik 3D (kedalaman punggung, kontur tulang, volume) dari depth map yang sudah disegmentasi .
  4. Prediksi: Model Machine Learning (MLP Regressor ) memprediksi skor BCS (skala 1-5) berdasarkan fitur tersebut.
  5. Output: Hasil ditampilkan di dashboard web/mobile yang user-friendly dengan mode offline fallback.
- **Value Utama:** Monitoring BCS yang objektif, akurat (target MAE < 0,25 ), real-time, non-invasif (mengurangi stres ternak ), dan dapat diakses (mendukung offline mode ).

---

## 3) Target User & Use Case
- **Persona Pengguna:** Peternak sapi perah (skala kecil-menengah di Lembang) , Dokter hewan , Peneliti.
- **Use Case Utama:** 
  - UC-1 (Capture): Peternak mengambil data RGB-D sapi menggunakan aplikasi di edge device yang terhubung depth camera.
  - UC-2 (Monitoring): Peternak/Dokter hewan melihat skor BCS terbaru (misal: 3.5) dan grafik tren historis BCS per sapi di dashboard untuk optimasi pakan. 
  - UC-3 (Alert): Sistem otomatis mengirim alert jika BCS sapi terdeteksi <2.5 (terlalu kurus) atau >4.0 (terlalu gemuk).

---

## 4) Core Features
| No | Fitur | Deskripsi Singkat | Prioritas (High/Med/Low) |
|----|-------|-------------------|---------------------------|
| 1  |    Akuisisi Data   RGB-D   |         Pengambilan gambar & depth map 16-bit via depth camera          |              High             |
| 2  |    Segmentasi Sapi   |        Isolasi tubuh sapi dari background (YOLOv8-Seg)           |              High             |
| 3  |    Prediksi Skor BCS   |         Ekstraksi fitur 3D & regresi skor kontinu (MLP Regressor)          |             High              |
| 4  |    Dashboard & Visualisasi   |         Tampilan web/mobile untuk grafik tren & alert system          |            High               |
| 5  |    Mode Offline   |          Kemampuan capture & simpan data di edge device saat koneksi buruk         |             Medium              |


---

## 5) System Design (Diagram Alur)
- **Arsitektur:** Hybrid Edge-Cloud. Pre-processing & capture di edge device (Android/Raspberry Pi). Inferensi model berat (YOLOv8, Regressor) & penyimpanan data historis di cloud.
- **Diagram Alur:** 

```mermaid
flowchart LR
  User[Peternak]

  subgraph Edge Device (Kandang)
    Mobile[Mobile App (Android)]
    Cam[Depth Camera (RealSense)]
    Cache[(Local Cache)]
    Mobile -->|Kontrol| Cam
    Cam -->|RGB-D| Mobile
    Mobile -->|Offline| Cache
  end

  subgraph Cloud (AWS/GCP)
    API[REST API (FastAPI)]
    ML[ML Inference Service<br/>1. YOLOv8-Seg<br/>2. Feature Extractor<br/>3. MLP Regressor]
    DB[(PostgreSQL)]
    Dash[Web Dashboard]
    API -->|Minta Prediksi| ML
    ML -->|Hasil Skor| API
    API -->|Simpan/Ambil Data| DB
    Dash -->|Query Data| API
  end

  User -->|Buka| Mobile
  Mobile -->|Sync Online| API
  API -->|Kirim Hasil| Mobile
  User -->|Buka| Dash
```

---

## 6) Tech Stack
| Layer | Tools/Framework | Catatan |
|-------|------------------|---------|
| Frontend | Android (Native/WebView) | Untuk capture & display di kandang. |
| Frontend | Web Dashboard (React/Vue.js) | Untuk visualisasi data historis & analitik. |
| Backend  | Python (FastAPI / Flask) | Menyediakan REST API untuk inference & data. |
| Database | PostgreSQL | Menyimpan metadata sapi dan skor BCS historis. |
| AI / ML  | PyTorch & Ultralytics | Untuk training & inference YOLOv8-Seg. |
| AI / ML  | Scikit-learn | Untuk MLPRegressor dan preprocessing. |
| AI / ML  | OpenCV, NumPy | Ekstraksi fitur morfometrik dari depth map 16-bit. |
| AI / ML  | Joblib | Penyimpanan model regressor (.pkl). |
| Cloud / Hosting | AWS (EC2, S3) atau GCP (Compute Engine) | Hosting backend API, database, dan model ML. |
| Version Control | Git | Manajemen versi dan kolaborasi. |
| Observability | CloudWatch / Datadog / Sentry | Monitoring uptime API dan error. |

---

## 7) Data Flow / Workflow
- **Data Flow:**
  1. Capture: Peternak mengambil gambar RGB + Depth (16-bit) via depth camera.
  2. Sync: Aplikasi mengirim data ke cloud API (jika online) atau menyimpan di cache lokal (jika offline).
  3. API Trigger: Backend API menerima data, memicu pipeline inferensi.
- **AI Pipeline:**
  1. Segmentasi: Gambar RGB di-input ke YOLOv8-Seg. Output: Mask tubuh sapi.
  2. Ekstraksi Fitur: Mask digunakan untuk memfilter depth map 16-bit. Fitur (misal: mean_depth, depth_variance, body_volume_est) dihitung dari depth map yang relevan .
  3. Prediksi: Vektor fitur di-input ke MLPRegressor (model .pkl). Output: Skor BCS kontinu (misal: 3.42).
  4. Post-processing: Skor disimpan ke database (PostgreSQL) dan dikirim kembali ke dashboard/mobile app.
- **AI Pipeline (jika ada):** preprocessing → inference → postprocessing
  - POST /api/v1/predict: Menerima (RGB, Depth), menjalankan pipeline, mengembalikan skor BCS.
  - GET /api/v1/cow/{id}/history: Mengembalikan data historis BCS untuk visualisasi grafik.

---

## 8) Security & Scalability
- **AuthN/AuthZ:** JWT (JSON Web Tokens) untuk mengamankan akses API dan dashboard.
- **Proteksi Data:** Enkripsi in-transit (SSL/TLS) untuk semua komunikasi API. Enkripsi at-rest untuk data sensitif di database
- **Scalability:** Arsitektur hybrid mengurangi beban bandwidth & latensi. Backend API di cloud dapat di-scale (auto-scaling, load balancer).
- **Monitoring:** Logging terpusat (misal: CloudWatch) untuk request API, latency inferensi, dan error sistem.
---

## 9) Timeline
| Phase | Deliverable | Estimasi Waktu | Status |
|-------|-------------|----------------|--------|
| 1     |      Analisis & Requirements       |       Minggu 5-6         |    Selesai    |
| 2     |       System Design & Arsitektur      |        Minggu 7-8        |    Selesai    |
| 3     |       Development & Implementasi      |        Minggu 9-11        |    In Progress    |
| 4     |       Testing & Validasi      |        Minggu 12-13        |    Next Up    |
| 5     |       Deployment & Evaluasi      |        Minggu 14        |    Belum    |

---

## 10) Next Steps
- **Pengujian/QA:** 
  - Model: Validasi model pada test set lapangan (target MAE < 0,25, R² > 0,75) .
  - Sistem: Performance test (target latensi < 5 detik online).
  - User: Usability Testing (SUS) dengan peternak (target skor > 70).
- **Deployment:** Pilot deployment sistem di suatu peternakan mitra di Jawa Barat.
- **Dokumentasi:** Menyiapkan materi user training (Bahasa Indonesia) dan dokumentasi teknis akhir.
