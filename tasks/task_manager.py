"""
DS-54 — Türkçe Ürün Yorumu Duygu Analizi: TF-IDF vs LSA Karşılaştırması

Bir e-ticaret / içerik platformunda data scientist'sin. Elinde gerçek
kullanıcılardan toplanmış yüzlerce Türkçe ürün yorumu var. Görevin sadece
"pozitif mi negatif mi" demek değil — aynı metni İKİ FARKLI şekilde sayısal
temsile çevirip (TF-IDF ve LSA), hangi temsilin Türkçe duygu analizinde daha
iyi çalıştığını ölçüp NEDEN diye sorgulamak.

  - Yöntem A — TF-IDF: metni SEYREK (sparse) bir vektöre çevirir. Her
    kelime/n-gram kendi boyutunu oluşturur, çoğu değer sıfırdır. Basit ama
    genelde küçük/orta veri setlerinde çok güçlü bir taban çizgisidir.

  - Yöntem B — LSA (TruncatedSVD): TF-IDF matrisini boyut indirgeme ile
    YOĞUN (dense), daha küçük bir uzaya sıkıştırır. Amaç, kelimelerin
    arkasındaki "gizil anlam"ı (latent semantic) yakalamak — bu, modern
    kelime embedding'lerinin klasik/lineer atasıdır.

Her fonksiyonun `raise NotImplementedError(...)` kısmını sil, docstring'deki
talimatlara göre kodu yaz. Testleri çalıştır, hepsi geçene kadar iterate et:
`python watch.py` veya `pytest tests/test_question.py -v`
"""

import re

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 1. Veriyi yükle
def load_data(path="data/tr_yorumlar.csv"):
    """
    Gerçek Türkçe ürün yorumlarını CSV'den okuyup DataFrame olarak döndür.

    Args:
        path: CSV dosya yolu (varsayılan: "data/tr_yorumlar.csv")

    Returns:
        pd.DataFrame: sütunlar ['yorum', 'duygu']
                      ('duygu' ∈ {'positive', 'negative'})

    İpucu:
    - pd.read_csv(path)
    """
    raise NotImplementedError("load_data fonksiyonunu tamamla")


# 2. Veriyi keşfet — temel istatistikler
def explore_data(df):
    """
    Veri setinin temel istatistiklerini çıkar.

    Args:
        df: load_data()'dan dönen DataFrame

    Returns:
        dict: {
            'n_rows': int,          # toplam satır sayısı
            'n_positive': int,      # 'positive' sayısı
            'n_negative': int,      # 'negative' sayısı
            'avg_length': float,    # ortalama kelime sayısı (yorum başına)
        }

    İpucu:
    - df['duygu'] == 'positive' ile filtrele, .sum() ile say
    - kelime sayısı: str(text).split() uzunluğu, .mean() ile ortalama
    """
    raise NotImplementedError("explore_data fonksiyonunu tamamla")


# 3. Metni temizle — Türkçe'ye duyarlı küçük harf + noktalama/rakam temizliği
def clean_text(text):
    """
    Türkçe'ye duyarlı metin temizliği yap:
    1. Türkçe'ye duyarlı küçük harf: Python'un standart .lower() metodu
       Türkçe için yanlış çalışır ("I".lower() → "i" verir ama Türkçe'de
       "I" → "ı" ve "İ" → "i" olmalıdır). Önce 'İ'→'i' ve 'I'→'ı' dönüşümünü
       ELLE yap, SONRA .lower() çağır.
    2. Noktalama işaretlerini ve rakamları temizle (Türkçe harfler
       çğıöşü korunmalı, geri kalanı boşlukla değiştirilmeli)
    3. Fazla boşlukları teke indir, baştaki/sondaki boşlukları kırp

    Örn: "Bu ÜRÜN çok İYİ!" → "bu ürün çok iyi"

    Args:
        text: str (ham yorum metni)

    Returns:
        str: temizlenmiş metin

    İpucu:
    - text = str(text).replace('İ', 'i').replace('I', 'ı').lower()
    - text = re.sub(r'[^a-zçğıöşü\\s]', ' ', text)
    - text = re.sub(r'\\s+', ' ', text).strip()
    """
    raise NotImplementedError("clean_text fonksiyonunu tamamla")


# 4. Seriye ön işleme uygula
def preprocess(series):
    """
    Bir pandas Series üzerindeki her yoruma clean_text uygula.

    Args:
        series: pd.Series (ham yorumlar, örn. df['yorum'])

    Returns:
        pd.Series: temizlenmiş yorumlar (aynı index korunur)

    İpucu:
    - series.apply(clean_text)
    """
    raise NotImplementedError("preprocess fonksiyonunu tamamla")


# 5. Eğitim/test ayrımı
def split_data(df):
    """
    Veriyi eğitim ve test setlerine ayır.

    1. X: preprocess(df['yorum']) — temizlenmiş yorum metni
    2. y: df['duygu'] üzerinden 0/1 kodlama — 'positive' → 1, 'negative' → 0
    3. train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    Args:
        df: load_data()'dan dönen DataFrame

    Returns:
        X_train, X_test, y_train, y_test

    İpucu:
    - y = (df['duygu'] == 'positive').astype(int)
    - from sklearn.model_selection import train_test_split
    """
    raise NotImplementedError("split_data fonksiyonunu tamamla")


# 6. Yöntem A — TF-IDF (seyrek/sparse temsil)
def train_tfidf(X_train, y_train):
    """
    TF-IDF + LogisticRegression pipeline'ı kur ve eğit.

    Yöntem A: metni SEYREK (sparse) bir vektöre çevirir — her kelime/n-gram
    kendi boyutunu oluşturur, çoğu boyut sıfırdır.

    Pipeline:
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2)))
        ('clf', LogisticRegression(max_iter=1000))

    Args:
        X_train: eğitim metinleri (temizlenmiş)
        y_train: eğitim etiketleri (0/1)

    Returns:
        sklearn.pipeline.Pipeline (eğitilmiş — .fit() çağrılmış)

    İpucu:
    - from sklearn.feature_extraction.text import TfidfVectorizer
    - from sklearn.linear_model import LogisticRegression
    - from sklearn.pipeline import Pipeline
    - pipe.fit(X_train, y_train); return pipe
    """
    raise NotImplementedError("train_tfidf fonksiyonunu tamamla")


# 7. Yöntem B — LSA / TruncatedSVD (yoğun/dense temsil)
def train_lsa(X_train, y_train):
    """
    TF-IDF + TruncatedSVD (LSA) + LogisticRegression pipeline'ı kur ve eğit.

    Yöntem B: TF-IDF'in seyrek matrisini TruncatedSVD ile 100 boyutlu
    YOĞUN (dense) bir uzaya indirger — kelimelerin arkasındaki "gizil anlam"
    (latent semantic) yapıyı yakalamayı hedefler. Bu, modern kelime
    embedding'lerinin klasik/lineer atasıdır.

    Pipeline:
        ('tfidf', TfidfVectorizer(max_features=5000))
        ('svd', TruncatedSVD(n_components=100, random_state=42))
        ('clf', LogisticRegression(max_iter=1000))

    Args:
        X_train: eğitim metinleri (temizlenmiş)
        y_train: eğitim etiketleri (0/1)

    Returns:
        sklearn.pipeline.Pipeline (eğitilmiş — .fit() çağrılmış)

    İpucu:
    - from sklearn.decomposition import TruncatedSVD
    - dikkat: TfidfVectorizer'a bu sefer ngram_range VERME (varsayılan kalsın)
    """
    raise NotImplementedError("train_lsa fonksiyonunu tamamla")


# 8. Modeli değerlendir
def evaluate(model, X_test, y_test):
    """
    Verilen model için test seti üzerinde accuracy/precision/recall/f1 hesapla.

    Args:
        model: eğitilmiş pipeline (train_tfidf veya train_lsa'dan)
        X_test: test metinleri
        y_test: test etiketleri (0/1)

    Returns:
        dict: {'accuracy': float, 'precision': float, 'recall': float, 'f1': float}

    İpucu:
    - from sklearn.metrics import accuracy_score, precision_score,
      recall_score, f1_score
    - y_pred = model.predict(X_test)
    """
    raise NotImplementedError("evaluate fonksiyonunu tamamla")


# 9. İki yöntemi karşılaştır
def compare_methods(tfidf_model, lsa_model, X_test, y_test):
    """
    TF-IDF (Yöntem A) ve LSA (Yöntem B) modellerinin test metriklerini
    yan yana karşılaştır.

    Args:
        tfidf_model: train_tfidf'ten dönen eğitilmiş pipeline
        lsa_model: train_lsa'dan dönen eğitilmiş pipeline
        X_test, y_test: test seti

    Returns:
        dict: {
            'tfidf': {'accuracy':..., 'precision':..., 'recall':..., 'f1':...},
            'lsa': {'accuracy':..., 'precision':..., 'recall':..., 'f1':...},
            'better_f1': 'tfidf' | 'lsa',   # hangi yöntemin f1'i daha yüksek
        }

    İpucu:
    - Her iki model için de evaluate() çağır
    - better_f1'i iki f1 değerini karşılaştırarak belirle
    """
    raise NotImplementedError("compare_methods fonksiyonunu tamamla")


# 10. Hata analizi — yanlış sınıflanan örnekler
def error_analysis(model, X_test, y_test, n=10):
    """
    Modelin yanlış sınıflandırdığı örnekleri incele — bu, "neden bu yöntem
    daha iyi/kötü çalıştı" sorusuna cevap ararken en değerli adımdır.

    Args:
        model: eğitilmiş pipeline
        X_test: pd.Series (temizlenmiş test yorumları)
        y_test: pd.Series (gerçek etiketler, 0/1)
        n: en fazla kaç yanlış örnek döndürülecek (varsayılan 10)

    Returns:
        list of dict: en fazla n tane, her biri şu formatta:
            {'yorum': str, 'gercek': int, 'tahmin': int}

    İpucu:
    - y_pred = model.predict(X_test)
    - X_test ve y_test'i numpy dizisine çevirip (np.asarray) sırayla gez
    - gercek != tahmin olan satırları topla, n'e ulaşınca dur
    """
    raise NotImplementedError("error_analysis fonksiyonunu tamamla")


if __name__ == "__main__":
    df = load_data()
    print("Veri istatistikleri:", explore_data(df))

    X_train, X_test, y_train, y_test = split_data(df)

    tfidf_model = train_tfidf(X_train, y_train)
    lsa_model = train_lsa(X_train, y_train)

    comparison = compare_methods(tfidf_model, lsa_model, X_test, y_test)
    print("Karşılaştırma:", comparison)

    errors = error_analysis(tfidf_model, X_test, y_test, n=5)
    print(f"Yanlış sınıflanan {len(errors)} örnek (TF-IDF):")
    for e in errors:
        print(" -", e)
