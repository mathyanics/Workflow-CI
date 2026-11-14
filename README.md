# Workflow-CI

Proyek machine learning end-to-end untuk klasifikasi risiko gagal bayar kredit (*Credit Default Risk*) dengan tracking eksperimen menggunakan **MLflow** dan otomasi CI/CD menggunakan **GitHub Actions**.

---

## 📁 Struktur Workspace

```text
Workflow-CI/
├── .github/
│   └── workflows/
│       └── train.yml               # GitHub Actions workflow untuk continuous integration
├── MLProject/
│   ├── data/
│   │   └── credit_default_preprocessed.csv
│   ├── conda.yaml                  # Spesifikasi environment Conda
│   ├── credit_default_preprocessed.csv
│   ├── MLproject                   # Spesifikasi proyek MLflow & entry points
│   ├── modelling.py                # Pipeline pelatihan baseline (autologging)
│   ├── modelling_tuning.py         # Pipeline hyperparameter tuning & visualisasi evaluasi
│   ├── README.md                   # Dokumentasi subproyek MLflow
│   └── requirements.txt            # Dependensi Python
└── README.md
```

---

## 🚀 Menjalankan Proyek Secara Lokal

### 1. Prasyarat
Pastikan Python 3.12 telah terpasang. Install dependensi yang diperlukan:

```bash
cd MLProject
pip install -r requirements.txt
```

### 2. Menjalankan Pelatihan Model Baseline
Jalankan entry point utama melalui MLflow CLI:

```bash
mlflow run . --env-manager=local
```

Atau jalankan skrip Python secara langsung:

```bash
python modelling.py --data credit_default_preprocessed.csv
```

### 3. Menjalankan Hyperparameter Tuning
Untuk melakukan tuning GridSearchCV dan mengekspor confusion matrix serta metrik evaluasi:

```bash
python modelling_tuning.py --data credit_default_preprocessed.csv
```

Atau via MLflow:

```bash
mlflow run . -e tuning --env-manager=local
```

### 4. Melihat Hasil Eksperimen di MLflow UI
```bash
mlflow ui
```
Buka browser di `http://127.0.0.1:5000` untuk melihat metrik, artefak, dan model yang tercatat.

---

## ⚙️ Integrasi CI/CD (GitHub Actions)

Alur kerja pada `.github/workflows/train.yml` akan secara otomatis:
1. Memvalidasi environment Python 3.12.7.
2. Menginstall dependensi dari `requirements.txt`.
3. Menjalankan MLflow Project untuk melatih model.
4. Memvalidasi bahwa run berhasil dan menghasilkan artefak model.
5. Mengunggah folder `mlruns/` sebagai GitHub Actions artifact.
