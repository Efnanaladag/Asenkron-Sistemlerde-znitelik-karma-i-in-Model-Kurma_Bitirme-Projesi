# Bitirme Projesi - EEG Control vs Non-Control (Single Subject, Multi Session)

Bu repo, Stieger2021 veri seti uzerinde tek subject ve coklu session kurulumunda
`control` vs `non-control` siniflandirma denemeleri icin hazirlanmistir.

## 1) Problem Tanimi

- Problem: `control` vs `non-control`
- Etiketleme:
  - `0 = ITI` (non-control)
  - `1 = feedback` (control)
- `cue` segmenti siniflandirmaya dahil edilmez.
- Metodolojik hedef: `pseudo-online`
- Veri kapsami: `tek subject`, `coklu session`
- Ana split: `LOSO-session` (Leave-One-Session-Out)
- Birincil metrik: `ROC-AUC`

## 2) Proje Yapisi

Ana dosyalar:

- Veri ve parse:
  - `load_stieger.py`
  - `parse_trials.py`
  - `build_labels.py`
- Pencereleme:
  - `windowing.py` (8-30 Hz)
  - `windowing_wideband.py` (0.5-50 Hz)
- Ozellik:
  - `features_bandpower.py`
- Egitim:
  - `train_baseline.py`
  - `train_csp_baseline.py`
  - `train_csp_5band_baseline.py`
  - `train_csp_5band_fs_baseline.py`
- Runner:
  - `run_cross_session.py`
  - `run_cross_session_csp.py`
  - `run_cross_session_csp_5band.py`
  - `run_cross_session_csp_5band_fs.py`
  - `run_batch_pipeline.py`

## 3) Pipeline Akisi

Session bazli veri uretimi:

1. `parse_trials.py <session_id>`
2. `build_labels.py <session_id>`
3. `windowing.py <session_id>` (veya wideband icin `windowing_wideband.py <session_id>`)
4. `features_bandpower.py <session_id>` (bandpower hattinda)

Cross-session egitim/degerlendirme:

- Bandpower + LDA:
  - `run_cross_session.py`
- Tek bant CSP + LDA:
  - `run_cross_session_csp.py`
- Canonical 5-band CSP + LDA:
  - `run_cross_session_csp_5band.py`
- Canonical 5-band CSP + FS + LDA:
  - `run_cross_session_csp_5band_fs.py`

Tum sessionlar icin batch veri uretimi:

- `run_batch_pipeline.py`

## 4) Kurulum

Not: Repo icinde `requirements.txt` bulunmuyorsa gerekli kutuphaneleri manuel kurmaniz gerekir.
Kod tabani asagidaki kutuphanelere dayanir:

- `numpy`
- `scikit-learn`
- `mne`
- `moabb`

Ornek (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy scikit-learn mne moabb
```

## 5) Hizli Baslangic (Ornek)

Tek session test:

```powershell
python parse_trials.py 1
python build_labels.py 1
python windowing.py 1
python features_bandpower.py 1
```

Tum sessionlar (subject config icinden okunur):

```powershell
python run_batch_pipeline.py
```

Bandpower cross-session:

```powershell
python run_cross_session.py
```

Tek bant CSP cross-session:

```powershell
python run_cross_session_csp.py
```

5-band CSP cross-session:

```powershell
python run_cross_session_csp_5band.py
python run_cross_session_csp_5band_fs.py
```

## 6) Cikti Klasorleri

Tum ciktilar `outputs/` altina yazilir.

Tipik alt klasorler:

- `outputs/trial_tables/`
- `outputs/label_tables/`
- `outputs/window_data/`
- `outputs/window_data_wideband/`
- `outputs/features/`
- `outputs/baseline_results/`
- `outputs/logs/`

## 7) Metodolojik Kurallar (Leakage Kirmizi Cizgi)

Bu projede su kurallar korunmalidir:

- Split oncesi supervised fit yapma.
- Session shuffle yapma.
- Scaler yalnizca train tarafinda fit edilmeli.
- CSP yalnizca train tarafinda fit edilmeli.
- Feature selection yalnizca train tarafinda fit edilmeli.

## 8) Konfigurasyon

Temel ayarlar `config.py` icindedir:

- `SUBJECT_ID`
- Frekans bantlari (`LOW_FREQ`, `HIGH_FREQ`, `BROAD_LOW_FREQ`, `BROAD_HIGH_FREQ`)
- `WINDOW_SIZE_SEC`, `STRIDE_SEC`
- `CSP_COMPONENTS`
- `RANDOM_SEED`

## 9) Notlar

- Bu repo su anda tek subject akisina odaklidir.
- Sonuc dosyalari ayni isimlere yazilabildigi icin farkli kosulari karsilastirirken
  hangi session kapsami ile uretildigini not etmek onemlidir.
