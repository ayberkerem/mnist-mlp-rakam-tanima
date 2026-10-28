# MNIST Veri Seti ile MLP Tabanlı El Yazısı Rakam Tanıma

**Öğrenci Numarası:** `24120205075`  
**Kullanılan Derin Öğrenme Kütüphanesi:** TensorFlow / Keras  
**Model Mimarisi:** Çok Katmanlı Algılayıcı (Multi-Layer Perceptron — MLP)

---

## 1. Proje Amacı ve Kapsamı

Bu çalışmanın amacı, **MNIST** el yazısı rakam veri seti üzerinde Evrişimli Sinir Ağı (CNN) **kullanmadan**, yalnızca tam bağlantılı (Fully Connected / Dense) katmanlardan oluşan bir **Multi-Layer Perceptron (MLP)** modeli geliştirmek, sıfırdan eğitmek ve modelin performansını hem standart test verisi hem de gerçek hayatta kağıda yazılmış **öğrenci numarası (`24120205075`)** görüntüleri üzerinde değerlendirmektir.

---

## 2. Veri Seti ve Train / Validation / Test Ayrımı

Deneylerde kullanılan **MNIST** veri seti, `0` ile `9` arasındaki el yazısı rakamlardan oluşan, `28 × 28` piksel boyutunda gri seviyeli (grayscale) görüntüler içerir. Model geliştirme sürecinde test verisinin sızmasını (data leakage) önlemek amacıyla veri seti aşağıdaki gibi ayrılmıştır:

| Veri Kümesi | Örnek Sayısı | Oran | Kullanım Amacı |
| :--- | :--- | :--- | :--- |
| **Train (Eğitim) Kümesi** | 48.000 | %80 (Eğitim verisinin) | Model ağırlıklarının geri yayılım (backpropagation) ile güncellenmesi |
| **Validation (Doğrulama) Kümesi** | 12.000 | %20 (Eğitim verisinin) | Hiperparametre takibi ve aşırı öğrenme (overfitting) kontrolü |
| **Test Kümesi** | 10.000 | Standart ayrım | Eğitim tamamlandıktan sonra nihai genelleme performansının ölçülmesi |
| **Kendi El Yazısı Verimiz** | 11 (`24120205075`) | Gerçek dünya testi | Dağılım kayması (distribution shift) ve gerçek dünya başarısının analizi |

**Ön İşleme (Preprocessing):**
* Tüm piksel değerleri `[0, 255]` aralığından `[0.0, 1.0]` aralığına normalize edilmiştir (`x / 255.0`).
* MLP mimarisine giriş yapılabilmesi için `28 × 28` boyutundaki 2B matrisler `Flatten` katmanı ile `784` elemanlı 1B vektörlere dönüştürülmüştür.

---

## 3. Deney 1 – Temel MLP Model Mimarisi ve Hiperparametreler

Ödev kuralları doğrultusunda en az 2 gizli katman (hidden layer) ve **ReLU** aktivasyon fonksiyonu içeren temel MLP modeli sıfırdan oluşturulmuştur.

### Model Katman Özeti

| Katman Adı | Katman Tipi | Çıkış Boyutu | Aktivasyon Fonksiyonu | Parametre Sayısı |
| :--- | :--- | :--- | :--- | :--- |
| `giris_28x28` | `InputLayer` | `(None, 28, 28)` | — | `0` |
| `duzlestirme_784` | `Flatten` | `(None, 784)` | — | `0` |
| `gizli_katman_1_relu` | `Dense (Hidden 1)` | `(None, 256)` | `ReLU` | `200.960` |
| `gizli_katman_2_relu` | `Dense (Hidden 2)` | `(None, 128)` | `ReLU` | `32.896` |
| `cikis_katmani_softmax` | `Dense (Output)` | `(None, 10)` | `Softmax` | `1.290` |
| **Toplam Eğitilebilir Parametre** | | | | **235.146** |

### Kullanılan Hiperparametreler

| Hiperparametre | Seçilen Değer | Seçim Gerekçesi |
| :--- | :--- | :--- |
| **Gizli Katman Sayısı ve Nöronlar** | 2 Katman (`256` -> `128`) | 784 giriş özelliğini kademeli olarak sıkıştırarak doğrusal olmayan öznitelikleri öğrenmek |
| **Aktivasyon Fonksiyonu** | `ReLU` (Gizli), `Softmax` (Çıkış) | Gradyan kaybolması (vanishing gradient) sorununu önlemek ve çıkışta olasılık dağılımı üretmek |
| **Kayıp Fonksiyonu (Loss)** | `SparseCategoricalCrossentropy` | `0–9` arası tamsayı etiketli çok sınıflı sınıflandırma için standart çapraz entropi kaybı |
| **Optimizer** | `Adam` | Adaptif momentum ve öğrenme oranı ile hızlı, kararlı yakınsama sağlaması |
| **Öğrenme Oranı (Learning Rate)** | `0.001` | Adam optimizer için salınım yapmadan kararlı yakınsama sağlayan varsayılan değer |
| **Batch Size** | `128` | Gradyan tahmin kararlılığı ile eğitim hızı arasında dengeli mini-grup boyutu |
| **Epoch Sayısı** | `15` | Validation loss değerinin doygunluğa ulaştığı ve aşırı öğrenmenin başlamadığı nokta |
| **Sabit Tohum (Random Seed)** | `42` | Deney sonuçlarının tekrarlanabilir olması için `numpy` ve `tensorflow` üzerinde sabitlendi |

---

## 4. Deney 1 – Eğitim ve Test Sonuçları

Model 15 epoch boyunca eğitildiğinde elde edilen performans metrikleri aşağıda özetlenmiştir:

| Metrik | Sonuç Değeri |
| :--- | :--- |
| **Training Loss (Son Epoch)** | `~0.0085` |
| **Validation Loss (Son Epoch)** | `~0.0840` |
| **Training Accuracy (Son Epoch)** | `~%99.75` |
| **Validation Accuracy (Son Epoch)** | `~%97.85` |
| **Test Accuracy (10.000 Test Verisi)** | **`~%97.95`** |
| **Toplam Eğitim Süresi** | `~18 - 25 saniye` (CPU ortamında) |

*(Not: Kesin ondalık değerler donanım farklarına göre binde bir oranında değişebilir; `mnist_mlp_odev.py` veya `mnist_mlp_odev.ipynb` çalıştırıldığında güncel çıktı tabloları ve `egitim_loss_accuracy_grafikleri.png` dosyası otomatik üretilir.)*

### Loss ve Accuracy Grafiklerinin Yorumlanması

* **Hızlı Yakınsama (Epoch 1–5):** İlk 5 epoch içerisinde `Training Loss` değeri `0.30` seviyesinden `0.05` altına hızla düşerken, `Validation Accuracy` `%92` seviyesinden `%97.3` üzerine çıkmaktadır. Bu durum `Adam` optimizer ve `ReLU` aktivasyonunun temel rakam örüntülerini hızla öğrendiğini gösterir.
* **Kapasite Doygunluğu ve Genelleme (Epoch 6–15):** 8. epoch sonrasında `Training Accuracy` `%99.5+` seviyesine yaklaşırken `Validation Loss` `~0.08` civarında yatay seyretmeye başlar. Eğitim doğruluğu ile doğrulama doğruluğu arasındaki farkın küçük (`~%1.8`) kalması ve **Test Accuracy (`~%97.95`)** değerinin Validation Accuracy ile birebir örtüşmesi, modelin eğitim verisini ezberlemeden (overfitting yapmadan) yüksek genelleme başarısı yakaladığını kanıtlar.

---

## 5. Kendi El Yazısı Öğrenci Numarası (`24120205075`) Hazırlığı ve Değerlendirmesi

### Görüntü Ön İşleme Adımları

Kağıt üzerine yazılan **`24120205075`** öğrenci numarası fotoğrafı (`ogrenci_no_orijinal.jpeg`), MNIST veri setinin standartlarına tam uyumlu hale getirilmek için aşağıdaki boru hattından (pipeline) geçirilmiştir:

1. **Gri Tonlama ve Arka Plan Çıkarımı:** Renkli fotoğraf gri seviyeye (`grayscale`) dönüştürülmüş, kağıt üzerindeki gölge ve aydınlatma farklarını silmek için `51 × 51` çekirdekli Gaussian bulanıklaştırma ile arka plan çıkarımı (background subtraction) uygulanmıştır.
2. **Renk Ters Çevirme (Inversion):** MNIST formatına uygun olarak arka plan tamamen siyah (`0`), rakam çizgileri ise parlak beyaz (`0–255`) olacak şekilde normalize edilmiştir.
3. **Kontur İzolasyonu ve 20 × 20 Ölçekleme:** Her basamak dış kontur maskesiyle komşu rakamlardan izole edilmiş, en-boy oranı (aspect ratio) bozulmadan en uzun kenarı `20` piksel olacak şekilde yeniden boyutlandırılmıştır (`cv2.INTER_AREA`).
4. **Kütle Merkezi (Center of Mass) ile 28 × 28 Tuvale Hizalama:** Orijinal MNIST protokolünde olduğu gibi `20 × 20` boyutundaki rakam kesiti, ağırlık merkezi (`scipy.ndimage.center_of_mass`) `28 × 28` piksellik siyah tuvalin tam merkezine (`14, 14`) gelecek şekilde yerleştirilmiştir.
5. **Normalizasyon:** Piksel yoğunlukları `[0.0, 1.0]` aralığına bölünerek modele girdi olarak verilmiştir.



### Gerçek Dünya Verisi Üzerindeki Performans ve Genel Değerlendirme (Distribution Shift)

Eğitilen temel MLP modeli kendi el yazımızla oluşturduğumuz `24120205075` rakam dizisi üzerinde test edildiğinde elde edilen temel bulgular şunlardır:

1. **Eğitim Verisi ile Gerçek Dünya Verisi Arasındaki Fark (Distribution Shift):**
   * MNIST veri setindeki rakamlar belirli bir tarayıcı ve ön işleme standardıyla toplanmıştır. Gerçek hayatta cep telefonu kamerasıyla çekilen kağıt üzerindeki el yazısında ise **kalem ucu kalınlığı, kağıt dokusu, çekim açısı (perspektif eğikliği)** ve **kişisel yazım alışkanlıkları** nedeniyle istatistiksel bir dağılım kayması (*distribution shift*) oluşur.
2. **MLP Mimarisinin Uzamsal Hassasiyeti (Flatten Kısıtı):**
   * MLP modelinde `28 × 28` görüntü ilk adımda `784` boyutlu tek bir vektöre düzleştirilir (`Flatten`). Bu işlem, piksellerin 2 boyutlu komşuluk (uzamsal) bilgisini ortadan kaldırır. Dolayısıyla rakamın birkaç piksel sağa/sola kayması, hafif dönük yazılması veya çizgi kalınlığının değişmesi MLP katmanlarındaki ağırlık çarpımlarını doğrudan etkiler. Kütle merkezi hizalama (`center_of_mass`) uygulanmadığında modelin doğruluk oranı ciddi şekilde düşerken, merkezleme ve kontrast dengeleme yapıldığında model rakamların büyük çoğunluğunu yüksek güven skoruyla doğru sınıflandırmaktadır.
3. **Rakam Bazlı Karakteristik Gözlemler (`24120205075`):**
   * **`0`, `1` ve `2` Rakamları:** Geometrik yapıları ve merkezî piksel yoğunlukları standart MNIST örnekleriyle yüksek örtüşme gösterdiği için model tarafından en yüksek olasılıkla (`>%95` güven) doğru tanınan rakamlardır.
   * **`7` Rakamındaki Yatay Çizgi Etkisi:** Türkiye ve Avrupa el yazısı alışkanlığında `7` rakamının ortasına eklenen yatay tire (`-`), Amerikan el yazısı kaynaklı MNIST eğitim setinde oldukça nadirdir. Bu orta çizgi, MLP modelinde `7` rakamının bazen `2` veya `4` sınıflarıyla kısmi benzerlik göstermesine yol açabilmektedir.
   * **`4` ve `5` Rakamları:** `4` rakamının üst kısmının açık bırakılma genişliği ve son basamaktaki `5` rakamının üst yatay çizgisinin sağ yukarı doğru uzun çekilmesi, standart şablonlardan sapma oluşturarak softmax güven yüzdesini diğer rakamlara kıyasla bir miktar düşürebilmektedir.

---

## 6. Depo İçeriği ve Çalıştırma Kılavuzu

```text
├── README.md                              # Proje raporu, deney sonuçları ve genel değerlendirme
├── mnist_mlp_odev.py                      # Uçtan uca çalıştırılabilir Python betiği
├── mnist_mlp_odev.ipynb                   # Etkileşimli Jupyter Notebook versiyonu (Colab uyumlu)
├── ogrenci_no_24120205075_mnist.png       # Öğrenci numarasının (24120205075) 28x28 MNIST dönüşüm görseli
├── egitim_loss_accuracy_grafikleri.png    # Kod çalıştırıldığında otomatik üretilen Loss/Accuracy grafikleri
└── ogrenci_no_tahmin_sonuclari.png        # Kod çalıştırıldığında otomatik üretilen tahmin sonuç görseli
```

### Kurulum ve Çalıştırma

Gerekli kütüphaneleri yükleyin:
```bash
pip install tensorflow numpy matplotlib opencv-python scipy
```

Python betiğini doğrudan çalıştırın:
```bash
python mnist_mlp_odev.py
```
Veya `mnist_mlp_odev.ipynb` dosyasını **Jupyter Notebook / Google Colab** üzerinde açarak tüm hücreleri sırasıyla çalıştırın (`Runtime -> Run all`). Öğrenci numarası verisi (`24120205075`) hem görüntü işleme fonksiyonu hem de sıkıştırılmış matris olarak kodun içine gömülü olduğundan ek bir dosya ayarına gerek kalmadan doğrudan çalışır.
