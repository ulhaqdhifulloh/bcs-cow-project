# Tech Stack – BCS Monitoring System (Dashboard)

Dokumen ini menjelaskan teknologi yang digunakan untuk pengembangan sistem dashboard Body Condition Score (BCS) Monitoring.

---

## 1. Frontend Tech Stack

### **1.1 Framework**
- **Laravel Blade**  
  Digunakan untuk merender halaman dashboard, daftar sapi, detail sapi, dan form login secara server-side rendering agar ringan dan cepat.

### **1.2 Interactivity Layer**
- **AJAX (jQuery / Fetch API)**  
  Digunakan untuk:
  - Memperbarui data sapi secara dinamis (filter, search, pagination)
  - Memuat grafik statistik tanpa reload halaman
  - Mengambil data dari API backend Laravel

### **1.3 Styling**
- **TailwindCSS**  
  Memberikan UI yang modern, responsif, dan ringan untuk dipakai dalam WebView Android.

### **1.4 Chart & Visualization**
- **Chart.js**  
  Digunakan untuk menampilkan grafik:
  - Tren BCS sapi
  - Perbandingan BCS per Sapi (by TAg)
  - Distribusi kategori (normal, kurus, obesitas)

### **1.5 Offline Support (opsional – untuk versi hybrid / WebView)**
- Service Worker  
- Cache API (IndexedDB)
- PWA Compatibility

---

## 2. Backend Tech Stack

### **2.1 Framework Backend**
- **Laravel 11 (PHP)**  
  Menangani:
  - Autentikasi (login)
  - Manajemen user (peternak)
  - Endpoint REST API untuk data sapi
  - Dashboard statistik
  - Penyimpanan hasil prediksi BCS
  - Sinkronisasi data dari edge device atau Python ML service

### **2.2 Database**
- **MySQL 8**  
  Menyimpan:
  - Data sapi (`cows`)
  - Riwayat BCS (`bcs_records`)
  - Data user (`users`)
  - Klasifikasi sapi berkebutuhan khusus
  - Log aktivitas sistem

### **2.3 API Design**
Backend menyediakan API:
- `POST /api/login`
- `GET /api/cows`
- `GET /api/cows/{id}`
- `GET /api/cows/filter`
- `GET /api/dashboard/statistics`
- `GET /api/cows/special-needs`
- `POST /api/bcs/upload` → menerima hasil prediksi dari ML server

### **2.4 Queue / Background Job (Opsional)**
- **Laravel Queue (Redis/Database driver)**  
  Untuk:
  - Menangani request prediksi BCS
  - Menyimpan laporan besar
  - Scheduled task untuk klasifikasi sapi khusus

---

## 3. Machine Learning Service (Server-Side)

### **3.1 Framework**
- **Python (FastAPI / Flask)**  
  Digunakan untuk:
  - Menjalankan YOLOv8-Seg untuk segmentasi sapi  
  - Mengambil depth features dari sensor  
  - Menjalankan model regresi (MLP) untuk prediksi BCS  
  - Mengirim hasil ke backend Laravel via API

### **3.2 Model Components**
- YOLOv8-Seg (segmentasi RGB-D)
- Depth feature extractor (morfometrik 3D)
- MLP Regressor (prediksi 1–5 BCS)

### **3.3 Komunikasi**
- **REST API** antara Laravel <-> Python:
  - Laravel meminta prediksi (opsional)
  - Python mengirim hasil prediksi ke Laravel
  - Data disimpan di MySQL

---

## 4. Infrastruktur Tambahan (Opsional)

### **4.1 Deployment**
- Nginx + PHP-FPM untuk Laravel
- Gunicorn/Uvicorn untuk Python ML service
- Docker (opsional)

### **4.2 Integrasi Edge Device**
Untuk kamera depth:
- Raspberry Pi / Jetson Nano
- Mengirim gambar / fitur ke ML server

---

## 5. Summary

| Layer            | Tools / Technology              |
|------------------|---------------------------------|
| Frontend         | Laravel Blade, AJAX, TailwindCSS, Chart.js |
| Backend          | Laravel 11, MySQL, REST API     |
| ML Service       | Python, FastAPI/Flask, YOLOv8-Seg, MLP |
| Optional Features| Queue, PWA, Docker, Offline Sync |

Tech stack ini stabil, ringan, scalable, dan cocok untuk peternak dengan literasi digital rendah serta kondisi jaringan tidak stabil.

