# YorumTR — Türkçe Ürün Yorumu Duygu Analizi

**Bitirme Projesi · Veri Bilimi** • Tahmini süre: ~4-6 saat • Değerlendirme: **eğitmen incelemesi (manuel)**

---

## 📨 Durum

Bir e-ticaret platformunda **data scientist** olarak çalışıyorsun. Ürün ekibinden şu istek geliyor:

> "Elimizde binlerce müşteri yorumu birikiyor ama hangisi memnun, hangisi şikayet, elle okumadan bilemiyoruz. Sana gerçek yorumlardan bir set bırakıyorum. Bunları otomatik **pozitif / negatif** ayıran bir model istiyorum. Ama tek bir yöntemle yetinme — **birkaç farklı yaklaşım dene, hangisi Türkçe'de daha iyi çalışıyor bana sayıyla göster ve neden öyle olduğunu açıkla.** Bir de modelin **nerede yanıldığına** bak; hangi yorumlarda hata yapıyor, bir örüntü var mı?"

Bu senin **bitirme projen**. Sana adım adım talimat vermiyoruz — gerçek bir işte de vermezler. Veriyi ve işin hedefini veriyoruz; çözümü bir data scientist gibi sen tasarlayacaksın.

## 🎯 İş hedefi

Türkçe ürün yorumlarını **pozitif/negatif** olarak sınıflandıran bir model kur. En az **iki farklı yaklaşım** dene, aralarında **adil bir karşılaştırma** yap ve modelin **hata yaptığı yerleri** analiz et. Sonuçları iş diliyle raporla.

## 📦 Elindeki veri

- `data/tr_yorumlar.csv` — gerçek Türkçe ürün yorumları, ~570 satır, dengeli (yaklaşık yarı yarıya pozitif/negatif).
- `data/veri_sozlugu.md` — kolon açıklamaları.

⚠️ **Bu gerçek dünya verisi.** Gerçek insanların yazdığı yorumlar — yazım hataları, kısaltmalar, konuşma diline özgü ifadeler var. Ayrıca Türkçe, metin işlemede İngilizce'den farklı davranır; **dile özgü tuzakları fark edip doğru ele almak** işin bir parçası (bunları senin için tek tek saymıyoruz).

## 🛠️ Başlarken

```bash
# 1. Bu repoyu fork'la, sonra kendi fork'unu klonla
git clone <senin-fork-url>
cd data-science-project-54

# 2. Sanal ortam (önerilir)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Önerilen kütüphaneleri kur
pip install -r requirements.txt

# 4. Analiz defterini aç ve başla
jupyter lab
```

Boş bir `analiz.ipynb` oluşturup keşifle başlayabilirsin.

## ✅ Bizden beklentiler

**Nasıl** çözeceğin sana kalmış (hangi kütüphane, hangi yöntem — özgürsün). Ama iyi bir teslimatta şunları görmek istiyoruz:

1. **Metni doğru hazırla** — Türkçe metni modele vermeden önce işlemen gerekiyor. Türkçe'ye özgü sorunları fark et ve gerekçesiyle çöz.
2. **En az 2 farklı yaklaşım dene ve adil karşılaştır** — metni sayıya çevirmenin/sınıflandırmanın birden fazla yolu var. En az ikisini kur, aynı test seti üzerinde **f1** ile kıyasla. (Adil karşılaştırma: aynı ayrım, aynı değerlendirme.)
3. **Doğru metriği seç ve gerekçelendir** — neden accuracy tek başına yeterli/yetersiz? Bu iş için hangi metrik daha anlamlı?
4. **Hata analizi yap** — modelin yanlış sınıfladığı yorumlara bak. Ortak bir örüntü var mı (ironi, çok kısa/bağlamsız yorumlar, hem övgü hem şikayet içeren cümleler)?
5. **İş diliyle raporla** — hangi yöntemi neden önerirsin, model nerede güvenilir nerede değil.

## 📤 Teslim edeceklerin

| Dosya | Ne olmalı |
|---|---|
| `analiz.ipynb` | Uçtan uca analiz defterin: keşif → ön işleme → modelleme → karşılaştırma → hata analizi, kararlarını markdown hücrelerinde **anlatarak**. |
| `train.py` | Yeniden çalıştırılabilir script: veriyi okur, modeli eğitir, karşılaştırma metriklerini ekrana basar. (`python train.py` diyip çalıştırabilmeliyiz.) |
| `RAPOR.md` | İş diliyle raporun. `RAPOR_SABLONU.md`'deki soruları cevapla. |

## 🧭 Nasıl değerlendirilecek

Otomatik test yok — projeni bir eğitmen inceleyip aşağıdaki rubriğe göre değerlendirecek:

| Boyut | Puan |
|---|---:|
| Metin ön işleme & Türkçe farkındalığı | 15 |
| En az 2 yaklaşım & adil karşılaştırma | 25 |
| Değerlendirme & metrik muhakemesi | 20 |
| Hata analizi & içgörü | 20 |
| Rapor & iletişim | 15 |
| Kod kalitesi & tekrar çalıştırılabilirlik | 5 |
| **Toplam** | **100** |

Geçmek için ~70/100 hedefle. **Puanın çoğu "en yüksek skoru bulmakta" değil; yöntemleri karşılaştırman, doğru metriği seçmen ve hatayı yorumlamanda.**

## 📈 Başarı hedefi

Katı bir eşik yok. İyi bir yaklaşım bu veride **f1 olarak ~0.80 ve üzerini** yakalar; oraya yaklaş. Ama asıl değerlendirilen: yöntemleri **nasıl karşılaştırdığın** ve sonucu nasıl **yorumladığın**. (İpucu: en gelişmiş yöntem her zaman kazanmayabilir — neden kazandığını/kaybettiğini tartışmak, kazanandan daha değerli.)

## 🚀 Nasıl gönderirsin

1. Bu repoyu **fork'la**, kendi hesabında çöz. Repo'nun **public** olduğundan emin ol.
2. Kaizu'da bu projede **"İncelet 🔍"** butonuna bas.
3. **GitHub repo linkini** ve **neler yaptığını** (yaklaşımın, kararların, özellikle bakmamızı istediğin yerler) yaz, gönder.
4. Eğitmenin projeni inceleyecek — yanıt **2-3 iş günü** sürebilir; sonucu ve geri bildirimi Kaizu'da göreceksin.

## 💡 Hatırlatmalar

- Kütüphane seçimi sana ait. `requirements.txt`'te önerilen bir başlangıç seti var; dilediğini ekleyebilirsin.
- Kod ve rapor **senin** olmalı — eğitmen sana yaklaşımını soracak.
- Amaç mükemmel bir model değil; **bir data scientist gibi düşünüp** yöntemleri kıyaslaman ve dürüstçe yorumlaman.

Başarılar 🚀
