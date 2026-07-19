# DS-54 — Türkçe Ürün Yorumu Duygu Analizi: TF-IDF vs LSA Karşılaştırması

**Modül**: NLP / Metin Sınıflandırma (Bitirme Projesi, **opsiyonel**) • **Süre**: 3-4 saat

## 🎯 Proje Senaryosu

Bir e-ticaret platformunda **data scientist** olarak çalışıyorsun. Elinde
**gerçek kullanıcılardan toplanmış yüzlerce Türkçe ürün yorumu** var — kurgusal
örnek cümleler değil, gerçek insanların gerçek ürünler ve içerikler hakkında
yazdığı, dengeli (yaklaşık yarı yarıya pozitif/negatif) bir veri seti.

Bu projede tek bir görevin yok. **İki farklı metin temsili yöntemini
karşılaştırıyorsun**:

- **Yöntem A — TF-IDF (seyrek/sparse temsil)**: metni, her kelime/kelime
  ikilisinin kendi boyutu olduğu, çoğu değeri sıfır olan **seyrek** bir
  vektöre çevirir. Basit ama şaşırtıcı derecede güçlü bir taban çizgisidir.
- **Yöntem B — LSA / TruncatedSVD (yoğun/dense temsil)**: TF-IDF matrisini
  **boyut indirgeme** (SVD) ile çok daha küçük, **yoğun** bir uzaya sıkıştırır.
  Amaç, kelimelerin arkasındaki "gizil anlamsal" (latent semantic) yapıyı
  yakalamaktır — bu, bugünün kelime **embedding**'lerinin klasik/lineer
  atasıdır.

Görevin: her iki yöntemi de eğitmek, **metriklerini yan yana koymak**, hangisi
daha iyi çalıştığını **f1 skoru** üzerinden belirlemek ve — en önemlisi —
modelin **hangi yorumlarda yanıldığını** inceleyerek **NEDEN** böyle bir
sonuç çıktığını yorumlamak.

> ⚠️ Bu proje sadece "sınıflandır" demiyor — **iki temsili karşılaştırıp
> hangisinin Türkçe'de daha iyi çalıştığını ve nedenini analiz ediyorsun.**
> Modül içindeki NLP projelerinden (Türkçe metin sınıflandırma, NLP capstone)
> farkı da tam olarak burada: onlar TEK bir yöntemi öğretiyordu, burada iki
> yöntemi ölçüp **hata analiziyle** yorumluyorsun.

## 📦 Proje Kurulumu

```bash
# Fork + clone
git clone <your-fork-url>
cd ds-54-turkce-yorum-analizi

# Virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate          # Windows

# Dependencies
pip install -r requirements.txt

# Auto test runner (dosya değişince çalışır)
python watch.py

# Manuel test
pytest tests/test_question.py -v
```

## 🔑 Kaizu Bağlantısı — `kaizu_config.py`

Skorunun Kaizu hesabına yazılması için **`kaizu_config.py`** dosyasını aç ve
**`USER_ID`** alanını kendi user_id'nle değiştir:

```python
USER_ID = 0      # ← Kaizu profilinden alıp buraya yaz
PROJECT_ID = 0   # ← Bu projeye ait, dokunma
```

User_id'ni Kaizu profilinden bulabilirsin (Profile → Settings → User ID).

Skor göndermek için tüm testleri toplu çalıştırmalısın:

```bash
python tests/test_question.py
```

Bu komut tüm testleri çalıştırır, **passed/total oranını otomatik Kaizu'ya
gönderir**. Geliştirme sırasında `pytest -v` kullanmaya devam edebilirsin
(skor göndermez).

## 📚 Veri Seti — Gerçek Türkçe Ürün Yorumları

Veri **repo'da hazır bulunur**: `data/tr_yorumlar.csv`.

- **~570 gerçek internet yorumu**, **dengeli** (yaklaşık yarı yarıya
  pozitif/negatif)
- Sütunlar: `yorum` (metin), `duygu` (`positive` / `negative`)
- Veri gerçek kaynaklardan derlendiği için ön elemeden geçirilmiştir; yine de
  gerçek dünya verisidir — kurgusal, temiz cümleler değil, gerçek yazım
  hataları, kısaltmalar ve konuşma diline özgü ifadeler içerir. Bu da onu
  **gerçekçi** bir NLP problemi yapar.

## 📋 Görevler (`tasks/task_manager.py`)

`task_manager.py` dosyasındaki fonksiyonların `raise NotImplementedError(...)`
kısmını sil, docstring'e göre kodu yaz.

1. **`load_data(path)`** — CSV'yi oku, DataFrame döndür
2. **`explore_data(df)`** — satır sayısı, sınıf dağılımı, ortalama yorum uzunluğu
3. **`clean_text(text)`** — Türkçe'ye duyarlı küçük harf + noktalama/rakam temizliği
4. **`preprocess(series)`** — Seriye `clean_text` uygula
5. **`split_data(df)`** — stratified train/test ayrımı (%80/%20)
6. **`train_tfidf(X_train, y_train)`** — **Yöntem A**: TF-IDF + LogisticRegression
7. **`train_lsa(X_train, y_train)`** — **Yöntem B**: TF-IDF + TruncatedSVD (LSA) + LogisticRegression
8. **`evaluate(model, X_test, y_test)`** — accuracy/precision/recall/f1
9. **`compare_methods(tfidf_model, lsa_model, X_test, y_test)`** — iki yöntemi yan yana karşılaştır
10. **`error_analysis(model, X_test, y_test, n)`** — yanlış sınıflanan örnekleri incele

## 🎓 Öğrenme Hedefleri

Bu projeyi bitirdiğinde:
- [x] **Türkçe'ye özgü** metin ön işleme yapabileceksin
- [x] **TF-IDF (seyrek temsil)** ile metni sayısal vektöre çevirebileceksin
- [x] **LSA / TruncatedSVD (yoğun temsil)** ile boyut indirgemeyi uygulayabileceksin
- [x] İki farklı metin temsilini **f1 skoru üzerinden karşılaştırabileceksin**
- [x] **Hata analizi** yaparak bir modelin neden yanıldığını yorumlayabileceksin
- [x] Sonuçları **raporlayıp yorumlayabileceksin** (aşağıdaki "Sonuçlarını Yorumla" bölümüne bak)

## 🧪 Testler

Test dosyası: `tests/test_question.py` (15 test)

Tümü pass olmalı:
- `load_data` yapısı (~570 satır, `['yorum','duygu']`)
- `explore_data` çıktısı (n_rows/n_positive/n_negative/avg_length, dengeli dağılım)
- `clean_text` küçük harf + noktalama/rakam temizliği + boşluk sadeleştirme
- `preprocess` seri üzerinde doğru çalışıyor
- `split_data` %80/%20 + stratify doğru
- `train_tfidf` / `train_lsa` pipeline döner, predict eder
- `evaluate` metrikleri 0-1 aralığında
- **TF-IDF f1 ≥ 0.80** (Türkçe sentiment'te TF-IDF genelde ~0.84 civarı verir)
- `compare_methods` her iki yöntem için metrik + `better_f1` geçerli anahtar
- `error_analysis` yanlış örnek listesi, her biri `yorum`+`gercek`+`tahmin` içerir, ≤n

## 📊 Sonuçlarını Yorumla (rapor bölümü)

`run_pipeline` benzeri bir akışı `if __name__ == "__main__":` bloğunda
çalıştırdığında şuna benzer bir çıktı göreceksin:

```
Karşılaştırma: {
  'tfidf': {'accuracy': ~0.83, 'precision': ~0.82, 'recall': ~0.86, 'f1': ~0.84},
  'lsa':   {'accuracy': ~0.76, 'precision': ~0.78, 'recall': ~0.74, 'f1': ~0.76},
  'better_f1': 'tfidf'
}
```

**Not**: LSA'nın TF-IDF'ten biraz düşük f1 vermesi **beklenen ve öğretici**
bir sonuçtur — boyut indirgeme (100 bileşene sıkıştırma) bir miktar bilgi
kaybına yol açar. Bu proje seni "TF-IDF her zaman kazanır" demeye değil,
**neden** kazandığını (Türkçe'nin ek/kök yapısında ayırt edici kelimelerin
seyrek temsilde daha net kalması, LSA'nın 100 boyuta sıkıştırırken bu ayrımı
bulanıklaştırması) düşünmeye itiyor.

`error_analysis` çıktısına bak: yanlış sınıflanan yorumlarda ironi, çok kısa
yorumlar ("çok çok fazla çökmüş burada" gibi bağlamsız ifadeler) veya karışık
duygu içeren cümleler (hem övgü hem eleştiri) öne çıkıyor mu? Bunu kısaca
kendi kelimelerinle yorumla.

## 💡 İpuçları

- **Türkçe lowercase**: önce `text.replace('İ','i').replace('I','ı')`, sonra
  `.lower()`. Sıra önemli — aksi halde "İyi" gibi kelimeler yanlış küçülür.
- **clean_text** regex: `re.sub(r'[^a-zçğıöşü\s]', ' ', text)` → Türkçe harf +
  boşluk dışını siler.
- **Pipeline ham string alır** — `TfidfVectorizer` içine zaten temizlenmiş
  metin veriyorsun (`split_data` bunu `preprocess` ile hallediyor), elle
  vektörize etmene gerek yok.
- **TF-IDF vs LSA farkı sadece bir satır**: `train_lsa` içinde `TfidfVectorizer`
  ile `LogisticRegression` arasına `TruncatedSVD(n_components=100)` ekleniyor.
  Bunun dışında pipeline mantığı aynı.
- **error_analysis**'ta `X_test`/`y_test` pandas Series olabilir — `np.asarray()`
  ile diziye çevirip index hizasını koruyarak gez.

## 🚫 Dikkat

- `tests/test_question.py` dosyasını **değiştirme**
- `random_state=42`, `test_size=0.2`, `max_features=5000`, `n_components=100`
  değerlerini değiştirme (testler ve eşikler bu değerlere göre kalibre edildi)
- `_solution/` klasörü yok (DB'de saklanır, dersin haftası geçince açılır)
- Dokunabileceğin dosyalar: `tasks/task_manager.py` (kodu yaz) +
  `kaizu_config.py` (sadece USER_ID)
