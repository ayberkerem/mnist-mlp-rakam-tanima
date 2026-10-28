# -*- coding: utf-8 -*-
"""
ÖDEV: MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma
Öğrenci No: 24120205075

Bu betik aşağıdaki adımları uçtan uca gerçekleştirir:
1. MNIST veri setini indirir ve yükler (60.000 eğitim + 10.000 test görüntüsü).
2. Eğitim verisini Train (48.000) ve Validation (12.000) olarak ayırır, Test (10.000) verisini
   yalnızca nihai değerlendirme için saklar.
3. Görüntüleri [0, 1] aralığına normalize eder ve 28x28 -> 784 boyutlu vektörlere düzleştirir.
4. En az 2 gizli katmanlı (256 ve 128 nöronlu, ReLU aktivasyonlu) temel bir MLP modeli
   oluşturur ve sıfırdan eğitir (CNN kullanılmaz).
5. Eğitim süresini, Training/Validation Loss ve Accuracy değerlerini ve Test Accuracy
   sonucunu raporlar; Loss ve Accuracy grafiklerini kaydeder.
6. Öğrenci numarası (24120205075) el yazısı görselini MNIST formatına (gri tonlama,
   arka plan/rakam renk uyumu, 28x28 boyutlandırma, kütle merkezi hizalama ve normalizasyon)
   dönüştürerek model üzerinde test eder ve tahmin sonuçlarını görselleştirir.
"""

import os
import time
import zlib
import base64
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import center_of_mass, shift
import tensorflow as tf
from tensorflow.keras import layers, models

# Tekrarlanabilirlik (Reproducibility) için sabit tohum (seed)
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ==============================================================================
# 1. VERİ SETİNİN YÜKLENMESİ VE TRAIN / VALIDATION / TEST AYRIMI
# ==============================================================================
def load_and_split_mnist(val_ratio=0.2):
    print("\n[1/5] MNIST Veri Seti Yükleniyor ve Hazırlanıyor...")
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalizasyon: [0, 255] -> [0.0, 1.0]
    x_train_full = x_train_full.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Train (48.000) ve Validation (12.000) ayrımı
    num_val = int(len(x_train_full) * val_ratio)
    indices = np.random.permutation(len(x_train_full))
    val_idx, train_idx = indices[:num_val], indices[num_val:]

    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    print(f"  - Train kümesi      : {x_train.shape[0]} örnek ({x_train.shape[1]}x{x_train.shape[2]})")
    print(f"  - Validation kümesi : {x_val.shape[0]} örnek ({x_val.shape[1]}x{x_val.shape[2]})")
    print(f"  - Test kümesi       : {x_test.shape[0]} örnek ({x_test.shape[1]}x{x_test.shape[2]})")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


# ==============================================================================
# 2. TEMEL MLP MODELİNİN OLUŞTURULMASI (DENEY 1)
# ==============================================================================
def build_mlp_model(learning_rate=0.001):
    """
    CNN katmanı kullanılmadan, yalnızca yoğun (Dense / Fully Connected) katmanlardan
    oluşan 2 gizli katmanlı Multi-Layer Perceptron (MLP) modeli.
    """
    model = models.Sequential([
        layers.Input(shape=(28, 28), name="giris_28x28"),
        layers.Flatten(name="duzlestirme_784"),
        layers.Dense(256, activation="relu", name="gizli_katman_1_relu"),
        layers.Dense(128, activation="relu", name="gizli_katman_2_relu"),
        layers.Dense(10, activation="softmax", name="cikis_katmani_softmax")
    ], name="MNIST_Temel_MLP")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==============================================================================
# 3. EĞİTİM SÜRECİ LOSS VE ACCURACY GRAFİKLERİNİN ÇİZİLMESİ
# ==============================================================================
def plot_training_history(history, save_path="egitim_loss_accuracy_grafikleri.png"):
    epochs = range(1, len(history.history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Grafiği
    ax1.plot(epochs, history.history["loss"], "o-", color="#1d4ed8", linewidth=2, label="Training Loss")
    ax1.plot(epochs, history.history["val_loss"], "s--", color="#dc2626", linewidth=2, label="Validation Loss")
    ax1.set_title("Eğitim ve Doğrulama Kaybı (Loss)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Accuracy Grafiği
    ax2.plot(epochs, [a * 100 for a in history.history["accuracy"]], "o-", color="#1d4ed8", linewidth=2, label="Training Accuracy (%)")
    ax2.plot(epochs, [a * 100 for a in history.history["val_accuracy"]], "s--", color="#16a34a", linewidth=2, label="Validation Accuracy (%)")
    ax2.set_title("Eğitim ve Doğrulama Doğruluğu (Accuracy)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    print(f"  - Eğitim grafikleri kaydedildi: {save_path}")


# ==============================================================================
# 4. KENDİ EL YAZISI ÖĞRENCİ NUMARASI GÖRSELLERİNİN HAZIRLANMASI (24120205075)
# ==============================================================================
# Yüklenen el yazısı fotoğrafından (24120205075) ön işleme ile elde edilen
# 11 adet 28x28 MNIST formatındaki rakam dizisi (dosya bulunamazsa yedek olarak kullanılır)
EMBEDDED_28X28_B64 = "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"
STUDENT_ID_DIGITS = [2, 4, 1, 2, 0, 2, 0, 5, 0, 7, 5]


def to_mnist_28x28(crop_diff):
    """Tek bir rakam kesitini 20x20 kutuya oranlayıp kütle merkezine göre 28x28 tuvale ortalar."""
    clean = np.where(crop_diff > 30, crop_diff, 0).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.dilate(clean, k, iterations=1)
    clean = cv2.normalize(clean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    ys, xs = np.where(clean > 20)
    if len(ys) > 0:
        clean = clean[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    h, w = clean.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(clean, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.float32)
    y_off = (28 - new_h) // 2
    x_off = (28 - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    cy, cx = center_of_mass(canvas)
    if not np.isnan(cy) and not np.isnan(cx):
        shift_y = np.clip(14.0 - cy, -4, 4)
        shift_x = np.clip(14.0 - cx, -4, 4)
        canvas = shift(canvas, shift=(shift_y, shift_x), mode="constant", cval=0.0)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def load_or_extract_student_digits(image_path="ogrenci_no_orijinal.jpeg"):
    """
    Eğer el yazısı fotoğrafı mevcutsa OpenCV ile gri tonlamaya çevirir, arka plan
    aydınlatmasını dengeler, renkleri ters çevirir (siyah zemin üzerine beyaz rakam)
    ve 28x28 piksele dönüştürür. Fotoğraf dizinde yoksa gömülü 28x28 diziyi yükler.
    """
    if os.path.exists(image_path):
        print(f"  - '{image_path}' dosyasından el yazısı rakamlar segmente ediliyor...")
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = cv2.GaussianBlur(gray, (51, 51), 0)
        diff = cv2.subtract(bg, gray)
        diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

        _, thresh = cv2.threshold(diff_norm, 42, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h > 25 and w > 8 and 100 < y < 400:
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [c], -1, 255, -1)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
                items.append((x, y, w, h, mask))

        items = sorted(items, key=lambda b: b[0])
        digits_28x28 = []
        for (x, y, w, h, mask) in items:
            pad = 6
            y1, y2 = max(0, y - pad), min(gray.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(gray.shape[1], x + w + pad)
            masked_diff = cv2.bitwise_and(diff_norm, diff_norm, mask=mask)[y1:y2, x1:x2]
            digits_28x28.append(to_mnist_28x28(masked_diff))

        if len(digits_28x28) == len(STUDENT_ID_DIGITS):
            return np.array(digits_28x28, dtype=np.uint8), np.array(STUDENT_ID_DIGITS)

    # Yedek: Gömülü 28x28 el yazısı öğrenci numarası verisini çöz
    raw = zlib.decompress(base64.b64decode(EMBEDDED_28X28_B64))
    digits_28x28 = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 28, 28)
    return digits_28x28, np.array(STUDENT_ID_DIGITS)


# ==============================================================================
# 5. ANA ÇALIŞTIRMA AKIŞI (EĞİTİM, TEST VE EL YAZISI DEĞERLENDİRMESİ)
# ==============================================================================
def main():
    # Hiperparametreler
    EPOCHS = 15
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    VAL_RATIO = 0.20

    print("=" * 72)
    print("MNIST VERİ SETİ İLE MLP TABANLI EL YAZISI RAKAM TANIMA (DENEY 1)")
    print("=" * 72)

    # 1. Veri setini yükle ve ayır
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_mnist(val_ratio=VAL_RATIO)

    # 2. Modeli oluştur
    print("\n[2/5] MLP Modeli Oluşturuluyor...")
    model = build_mlp_model(learning_rate=LEARNING_RATE)
    model.summary()

    # 3. Modeli eğit ve süreyi ölç
    print("\n[3/5] Model Eğitimi Başlıyor...")
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2
    )
    training_time = time.time() - start_time

    # 4. Sonuçları hesapla ve raporla
    print("\n[4/5] Model Performansı Değerlendiriliyor...")
    train_loss = history.history["loss"][-1]
    train_acc = history.history["accuracy"][-1]
    val_loss = history.history["val_loss"][-1]
    val_acc = history.history["val_accuracy"][-1]

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    print("\n" + "=" * 50)
    print("DENEY 1 - PERFORMANS METRİKLERİ ÖZETİ")
    print("=" * 50)
    print(f"  Training Loss       : {train_loss:.4f}")
    print(f"  Validation Loss     : {val_loss:.4f}")
    print(f"  Training Accuracy   : %{train_acc * 100:.2f}")
    print(f"  Validation Accuracy : %{val_acc * 100:.2f}")
    print(f"  Test Accuracy       : %{test_acc * 100:.2f} (Test Loss: {test_loss:.4f})")
    print(f"  Toplam Eğitim Süresi: {training_time:.2f} saniye")
    print("=" * 50)

    plot_training_history(history, save_path="egitim_loss_accuracy_grafikleri.png")

    # 5. Kendi el yazısı öğrenci numarası (24120205075) üzerinde değerlendirme
    print("\n[5/5] Kendi El Yazısı Öğrenci Numarası (24120205075) Test Ediliyor...")
    digits_u8, true_labels = load_or_extract_student_digits()

    # Normalizasyon: [0, 255] -> [0.0, 1.0]
    digits_norm = digits_u8.astype("float32") / 255.0

    probs = model.predict(digits_norm, verbose=0)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1) * 100.0

    correct_count = int(np.sum(preds == true_labels))
    custom_acc = (correct_count / len(true_labels)) * 100.0

    print(f"\n  Öğrenci Numarası Gerçek : {''.join(map(str, true_labels))}")
    print(f"  Modelin Tahmin Dizisi   : {''.join(map(str, preds))}")
    print(f"  Doğru Tahmin Oranı      : {correct_count} / {len(true_labels)} (%{custom_acc:.2f})\n")

    for idx, (t, p, c) in enumerate(zip(true_labels, preds, confs), start=1):
        durum = "DOĞRU" if t == p else "YANLIŞ"
        print(f"  Basamak {idx:2d} | Gerçek: {t} | Tahmin: {p} | Güven: %{c:5.1f} | [{durum}]")

    # Tahmin sonuçlarını görselleştir ve kaydet
    fig, axes = plt.subplots(1, len(true_labels), figsize=(15, 2.6))
    for i in range(len(true_labels)):
        axes[i].imshow(digits_u8[i], cmap="gray", vmin=0, vmax=255)
        color = "#16a34a" if preds[i] == true_labels[i] else "#dc2626"
        axes[i].set_title(
            f"Gerçek: {true_labels[i]}\nTahmin: {preds[i]}\n(%{confs[i]:.0f})",
            fontsize=9, fontweight="bold", color=color
        )
        axes[i].axis("off")

    plt.suptitle(
        f"El Yazısı Öğrenci Numarası (24120205075) MLP Tahmin Sonuçları — Başarı: {correct_count}/{len(true_labels)} (%{custom_acc:.1f})",
        fontsize=11, fontweight="bold", y=1.05
    )
    plt.tight_layout()
    plt.savefig("ogrenci_no_tahmin_sonuclari.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("\n  - El yazısı tahmin görseli kaydedildi: ogrenci_no_tahmin_sonuclari.png")


if __name__ == "__main__":
    main()