# MLflow Project: Credit Default Risk

Subproyek ini berisi kode pelatihan, hyperparameter tuning, dan spesifikasi MLflow Project untuk pemodelan risiko gagal bayar (*credit default*).

---

## 📋 File & Komponen

- `MLproject`: File konfigurasi standar MLflow yang mendefinisikan entry points (`main` dan `tuning`).
- `modelling.py`: Skrip pelatihan model baseline menggunakan Logistic Regression dengan `mlflow.sklearn.autolog()`.
- `modelling_tuning.py`: Skrip tuning hyperparameter dengan `GridSearchCV`, menyimpan matriks konfusi dan metrik ke `model_artifacts/`.
- `requirements.txt`: Daftar pustaka Python yang dibutuhkan.
- `conda.yaml`: Konfigurasi environment untuk manajemen environment via Conda.
- `credit_default_preprocessed.csv`: Dataset yang telah dipreproses untuk input pelatihan.

---

## 🏃 Cara Menjalankan

### Pelatihan Baseline (Default)
```bash
mlflow run . --env-manager=local
```

### Hyperparameter Tuning
```bash
mlflow run . -e tuning --env-manager=local
```
