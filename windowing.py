# windowing.py

import csv
import os
import sys

import numpy as np

import config
from load_stieger import load_stieger_subject


# =========================================================
# 1) BASİT AYARLAR
# =========================================================

# Session bilgisini artık sabit vermek yerine terminalden alacağız.
# Böylece aynı kodu bütün session'lar için tekrar tekrar kullanabiliriz.
# Örnek:
# python windowing.py 1
# python windowing.py 2

# Kılavuza göre ilk sürüm ayarları
TARGET_SFREQ = config.TARGET_SFREQ   # 1000 Hz'den 250 Hz'e resample yapacağız, böylece işlem süresini azaltır ve genellikle 250 Hz, 30 Hz'ye kadar olan beyin dalgalarını yakalamak için yeterlidir.
LOW_FREQ = config.LOW_FREQ           # 8 Hz'in altındaki frekanslar genellikle delta ve theta bantlarına denk gelir, bu bantlar motor kontrol durumunu yakalamak için çok kritik değildir. Bu yüzden 8 Hz'in altını filtreleyerek, daha dar bir bantta çalışacağız.
HIGH_FREQ = config.HIGH_FREQ         # 30 Hz'in üzerindeki frekanslar genellikle beta ve gamma bantlarına denk gelir, bu bantlar motor kontrol durumunu yakalamak için çok kritik değildir. Bu yüzden 30 Hz'in üzerini filtreleyerek, daha dar bir bantta çalışacağız.
WINDOW_SIZE_SEC = config.WINDOW_SIZE_SEC   # 2.0
STRIDE_SEC = config.STRIDE_SEC             # 1.0

# İlk sürümde motor bölge ile başlıyoruz
MOTOR_CHANNELS = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C3", "C1", "Cz", "C2", "C4",
    "CP3", "CP1", "CPz", "CP2", "CP4"
]  # Bu kanallar, beynin motor kontrolüyle ilişkili bölgelerden seçilmiştir. Bu kanalları kullanarak, modelin motor kontrol durumunu daha iyi öğrenmesini hedefliyoruz.


# =========================================================
# 2) LABEL TABLOSUNU OKU
# =========================================================

def get_session_from_cli():
    """
    Terminalden session numarasını alır.

    Örnek:
    python windowing.py 1
    """
    if len(sys.argv) < 2:
        raise ValueError("Kullanım: python windowing.py <session_id>")

    session_name = str(sys.argv[1])
    return session_name


def read_label_table(session_name):
    """
    build_labels.py çıktısını okur.
    """
    file_path = os.path.join(
        config.OUTPUT_DIR,
        "label_tables",
        f"session_{session_name}_labels.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Label tablosu bulunamadı: {file_path}")

    rows = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        raise ValueError("Label tablosu boş görünüyor.")

    return rows


# =========================================================
# 3) SADE SESSION RAW ÇEKME
# =========================================================

def get_session_raw(subject_data, session_name):
    """
    İstenen session içindeki ilk run'ın raw nesnesini döndürür.
    """
    session_name = str(session_name)

    if session_name not in subject_data:
        raise ValueError(f"Session {session_name} bulunamadı.")

    runs = subject_data[session_name]
    run_names = list(runs.keys())

    if len(run_names) == 0:
        raise ValueError(f"Session {session_name} içinde run yok.")

    run_name = run_names[0]
    raw = runs[run_name]

    return raw, run_name


# =========================================================
# 4) PREPROCESSING
# =========================================================

def preprocess_raw(raw):
    """
    İlk baseline için sade preprocessing:

    - motor kanalları seç
    - average reference uygula
    - 250 Hz'e resample et
    - 8-30 Hz filtre uygula

    Not:
    Bu sprintte notch zorunlu değil.
    """
    raw_proc = raw.copy()

    # Sadece var olan motor kanalları seç
    existing_channels = [ch for ch in MOTOR_CHANNELS if ch in raw_proc.ch_names]

    if len(existing_channels) == 0:
        raise ValueError("Motor kanal listesinde raw içinde bulunan kanal yok.")

    raw_proc.pick(existing_channels)

    # Basit ve anlaşılır referans
    raw_proc.set_eeg_reference("average", projection=False)

    # Tüm analizler resample sonrası ilerlesin
    raw_proc.resample(TARGET_SFREQ)

    # İlk baseline için dar bant
    raw_proc.filter(LOW_FREQ, HIGH_FREQ, verbose=False)

    return raw_proc


# =========================================================
# 5) PENCERE SAYISI / SINIR KONTROLÜ
# =========================================================

def segment_too_short(start_sec, end_sec):
    """
    Segment pencere çıkarmaya yetiyor mu kontrol eder.
    """
    return (end_sec - start_sec) < WINDOW_SIZE_SEC


# =========================================================
# 6) GERÇEK EEG PENCERELERİNİ ÇIKAR
# =========================================================

def extract_windows_from_segment(raw_proc, segment_row, start_window_id):
    """
    Tek bir label segmentinden gerçek EEG pencereleri çıkarır.

    Çıktı:
    - windows_data: numpy dizileri listesi
    - windows_meta: metadata sözlükleri listesi
    - dropped_row: segment kısa ise bilgi satırı
    - next_window_id: sıradaki pencere id
    """
    windows_data = []
    windows_meta = []

    segment_id = int(segment_row["segment_id"])
    trial_id = int(segment_row["trial_id"])
    session_id = str(segment_row["session_id"])
    run_id = str(segment_row["run_id"])
    source_segment = str(segment_row["source_segment"])
    label = int(segment_row["label"])
    cue_label = str(segment_row["cue_label"])

    start_sec = float(segment_row["start_sec"])
    end_sec = float(segment_row["end_sec"])
    duration_sec = float(segment_row["duration_sec"])

    if segment_too_short(start_sec, end_sec):
        dropped_row = {
            "segment_id": segment_id,
            "trial_id": trial_id,
            "session_id": session_id,
            "run_id": run_id,
            "source_segment": source_segment,
            "label": label,
            "cue_label": cue_label,
            "segment_start_sec": start_sec,
            "segment_end_sec": end_sec,
            "segment_duration_sec": duration_sec,
            "reason": "segment_too_short_for_window"
        }
        return windows_data, windows_meta, dropped_row, start_window_id

    sfreq = raw_proc.info["sfreq"]
    window_id = start_window_id
    current_start = start_sec

    while current_start + WINDOW_SIZE_SEC <= end_sec + 1e-9:
        current_end = current_start + WINDOW_SIZE_SEC

        start_sample = int(round(current_start * sfreq))
        end_sample = int(round(current_end * sfreq))

        # Gerçek EEG penceresi: shape = (n_channels, n_samples)
        window_data = raw_proc.get_data(start=start_sample, stop=end_sample)

        expected_n_samples = int(round(WINDOW_SIZE_SEC * sfreq))

        # Güvenlik kontrolü
        if window_data.shape[1] != expected_n_samples:
            current_start += STRIDE_SEC
            continue

        windows_data.append(window_data)

        meta_row = {
            "window_id": window_id,
            "segment_id": segment_id,
            "trial_id": trial_id,
            "session_id": session_id,
            "run_id": run_id,
            "source_segment": source_segment,
            "label": label,
            "cue_label": cue_label,
            "window_start_sec": round(current_start, 4),
            "window_end_sec": round(current_end, 4),
            "window_duration_sec": WINDOW_SIZE_SEC,
            "window_start_sample": start_sample,
            "window_end_sample": end_sample
        }

        windows_meta.append(meta_row)

        window_id += 1
        current_start += STRIDE_SEC

    return windows_data, windows_meta, None, window_id


# =========================================================
# 7) TÜM SEGMENTLER İÇİN
# =========================================================

def build_all_windows(raw_proc, label_rows):
    """
    Tüm label segmentlerinden gerçek EEG pencereleri çıkarır.
    """
    all_window_data = []
    all_window_meta = []
    dropped_rows = []

    next_window_id = 1

    for row in label_rows:
        window_data_list, window_meta_list, dropped_row, next_window_id = extract_windows_from_segment(
            raw_proc,
            row,
            next_window_id
        )  # Her segmentten çıkan pencere verisi ve meta bilgisini büyük listelere eklendi. Eğer segment kısa ise dropped_row bilgisi de kaydedildi

        all_window_data.extend(window_data_list)
        all_window_meta.extend(window_meta_list)

        if dropped_row is not None:
            dropped_rows.append(dropped_row)

    if len(all_window_data) == 0:
        raise ValueError("Hiç pencere üretilemedi.")

    X = np.stack(all_window_data, axis=0)   # (n_windows, n_channels, n_samples)
    y = np.array([row["label"] for row in all_window_meta], dtype=int)

    return X, y, all_window_meta, dropped_rows


# =========================================================
# 8) KISA ÖZET
# =========================================================

def print_window_preview(window_meta, max_rows=10):
    """
    İlk birkaç pencereyi ekrana basar.
    """
    print("\n===== İLK PENCERELER =====")

    limit = min(max_rows, len(window_meta))

    for i in range(limit):
        row = window_meta[i]
        print(
            f"window_id={row['window_id']} | "
            f"trial_id={row['trial_id']} | "
            f"type={row['source_segment']} | "
            f"label={row['label']} | "
            f"start={row['window_start_sec']} | "
            f"end={row['window_end_sec']}"
        )


def print_window_summary(X, y, dropped_rows):
    """
    Genel pencere özetini ekrana basar.
    """
    n_label_0 = int(np.sum(y == 0))
    n_label_1 = int(np.sum(y == 1))

    print("\n===== WINDOW ÖZETİ =====")
    print("X shape:", X.shape)
    print("Toplam pencere sayısı:", len(y))
    print("Label 0 pencere sayısı:", n_label_0)
    print("Label 1 pencere sayısı:", n_label_1)
    print("Dışlanan kısa segment sayısı:", len(dropped_rows))


# =========================================================
# 9) KAYIT
# =========================================================

def save_metadata_csv(window_meta, session_name):
    """
    Pencere metadata'sını CSV olarak kaydeder.
    """
    save_dir = os.path.join(config.OUTPUT_DIR, "window_data")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_window_metadata.csv")

    fieldnames = list(window_meta[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(window_meta)

    print(f"Metadata kaydedildi: {save_path}")


def save_dropped_segments_csv(dropped_rows, session_name):  # AI önerisi
    """
    Kısa kaldığı için dışlanan segmentleri kaydeder.
    """
    save_dir = os.path.join(config.OUTPUT_DIR, "window_data")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_dropped_segments.csv")

    if len(dropped_rows) == 0:
        fieldnames = [
            "segment_id", "trial_id", "session_id", "run_id", "source_segment",
            "label", "cue_label", "segment_start_sec", "segment_end_sec",
            "segment_duration_sec", "reason"
        ]
    else:
        fieldnames = list(dropped_rows[0].keys())

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dropped_rows)

    print(f"Dropped segmentler kaydedildi: {save_path}")


def save_npz(X, y, session_name):
    """
    Gerçek EEG pencerelerini npz dosyası olarak kaydeder.
    """
    save_dir = os.path.join(config.OUTPUT_DIR, "window_data")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"session_{session_name}_windows.npz")

    np.savez_compressed(save_path, X=X, y=y)

    print(f"EEG pencere verisi kaydedildi: {save_path}")


# =========================================================
# 10) ANA AKIŞ
# =========================================================

def main():
    subject_id = config.SUBJECT_ID
    session_name = get_session_from_cli()

    print(f"Subject {subject_id} yükleniyor...")
    subject_data = load_stieger_subject(subject_id)

    print(f"Session {session_name} seçiliyor...")
    raw, run_name = get_session_raw(subject_data, session_name)

    # Subject Data:
    # Subject içindeki tüm session ve run'ların raw nesnelerini içeren büyük bir sözlük.
    # Artık sadece ilgilendiğimiz session'ın raw'una ihtiyacımız var,
    # bu yüzden büyük sözlüğe artık ihtiyacımız yok ve belleği boşaltmak için silmek iyi olur.
    del subject_data

    print("Label tablosu okunuyor...")
    label_rows = read_label_table(session_name)

    print("Preprocessing uygulanıyor...")
    raw_proc = preprocess_raw(raw)

    print("Gerçek EEG pencereleri çıkarılıyor...")
    X, y, window_meta, dropped_rows = build_all_windows(raw_proc, label_rows)

    print_window_preview(window_meta)
    print_window_summary(X, y, dropped_rows)

    save_npz(X, y, session_name)
    save_metadata_csv(window_meta, session_name)
    save_dropped_segments_csv(dropped_rows, session_name)

    print("\nWindowing tamamlandı.")


if __name__ == "__main__":
    main()