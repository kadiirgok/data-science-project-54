"""
YorumTR — Türkçe Ürün Yorumu Duygu Analizi
Yeniden çalıştırılabilir eğitim scripti.

Kullanım:
    python train.py

Beklenen dosya yapısı:
    data/tr_yorumlar.csv   (kolonlar: yorum, duygu)
"""

import re
import time
import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

import zeyrek
from sentence_transformers import SentenceTransformer


# Sabit Değişkenler 
DATA_PATH = "tr_yorumlar.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
EMBEDDING_MODEL_NAME = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"

STANDART_STOPWORDS = {
    "ve", "veya", "da", "de", "ile", "için", "çünkü", "bir", "bu", "şu", "o",
    "ki", "mi", "mı", "mu", "mü", "gibi", "hem", "ise", "diye", "en", "daha",
    "böyle", "şöyle", "her", "hiç", "bazı", "tüm", "hep", "kendi",
    "ama", "fakat", "lakin", "ancak", "oysa", "halbuki", "yine", "rağmen",
}
KORUNACAK_ZITLIK_BAGLACLARI = {
    "ama", "fakat", "lakin", "ancak", "oysa", "halbuki", "yine", "rağmen", "oysaki",
}
TEMIZLIK_LISTESI = STANDART_STOPWORDS - KORUNACAK_ZITLIK_BAGLACLARI

HARF_HARITASI = {"İ": "i", "I": "ı"}
NOKTALAMA_REGEX = r'[.,!?;:"\'()\[\]{}–\-—<>‘“’”（）«»*_+~`|=#]+'

logging.getLogger("zeyrek").setLevel(logging.ERROR)
_analyzer = zeyrek.MorphAnalyzer()  # bir kere yükle, tekrar kullan


#Data için ağır temizlik 
def preprocess(text: str) -> str:
    # Adım 1: Türkçe uyumlu küçük harfe çevirme (İ->i, I->ı sorunundan kaçınmak için)
    metin = text
    for buyuk, kucuk in HARF_HARITASI.items():
        metin = metin.replace(buyuk, kucuk)
    metin = metin.lower()

    # Adım 2: Tekrarlayan harfleri normalize et ("kiiiiii" -> "ki")
    metin = re.sub(r"(\w)\1{2,}", r"\1", metin)

    # Adım 3: Noktalama temizliği (URL/email korumalı)
    urls = re.findall(r"https?://\S+|www\.\S+", metin)
    emails = re.findall(r"\S+@\S+", metin)
    for i, url in enumerate(urls):
        metin = metin.replace(url, f" [URL_{i}] ")
    for i, email in enumerate(emails):
        metin = metin.replace(email, f" [EMAIL_{i}] ")

    metin = re.sub(r"(?<!\w)" + NOKTALAMA_REGEX, " ", metin)
    metin = re.sub(NOKTALAMA_REGEX + r"(?!\w)", " ", metin)
    metin = re.sub(r"\s+" + NOKTALAMA_REGEX + r"\s+", " ", metin)

    for i, url in enumerate(urls):
        metin = metin.replace(f" [URL_{i}] ", url)
    for i, email in enumerate(emails):
        metin = metin.replace(f" [EMAIL_{i}] ", email)

    metin = re.sub(r"\s+", " ", metin).strip()

    # Adım 4: Zeyrek ile lemmatization + stopword filtreleme
    analiz_sonuclari = _analyzer.analyze(metin)
    kok_kelimeler = []

    for kelime_analizi in analiz_sonuclari:
        if not kelime_analizi:
            continue

        en_olasi_analiz = kelime_analizi[0]
        orijinal_kelime = en_olasi_analiz.word.lower()
        kelime_koku = getattr(en_olasi_analiz, "lemma", en_olasi_analiz).lower()

        # Zıtlık bağlaçlarında zeyrek'e güvenme (örn. "ama" -> "âmâ" hatası), orijinali kullan
        if orijinal_kelime in KORUNACAK_ZITLIK_BAGLACLARI:
            kelime_koku = orijinal_kelime

        # Zeyrek tanıyamadıysa (Unk), orijinali koru (bilgi kaybını önlemek için)
        if kelime_koku == "unk":
            kelime_koku = orijinal_kelime

        if not kelime_koku.isalpha():
            continue

        if kelime_koku not in TEMIZLIK_LISTESI:
            kok_kelimeler.append(kelime_koku)

    return " ".join(kok_kelimeler)


#Ön işleme 
def hafif_temizle(text: str) -> str:
    metin = text
    for buyuk, kucuk in HARF_HARITASI.items():
        metin = metin.replace(buyuk, kucuk)
    metin = metin.lower()
    metin = re.sub(r"(\w)\1{2,}", r"\1", metin)
    metin = re.sub(r"\s+", " ", metin).strip()
    return metin


#Akıs main kısmı
def main():
    print("Veri okunuyor...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {df.shape[0]} satır, {df.shape[1]} kolon")

    print("\nÖn işleme uygulanıyor (TF-IDF için ağır, embedding için hafif)...")
    t0 = time.time()
    df["yorum_temiz"] = df["yorum"].apply(preprocess)
    df["yorum_hafif"] = df["yorum"].apply(hafif_temizle)
    print(f"  Süre: {time.time() - t0:.1f} sn")

    # Tek bir split, her iki yaklaşımda da AYNI train/test bölünmesi kullanılıyor
    # (adil karşılaştırma için şart)
    X_temiz_train, X_temiz_test, y_train, y_test = train_test_split(
        df["yorum_temiz"], df["duygu"],
        test_size=TEST_SIZE, stratify=df["duygu"], random_state=RANDOM_STATE,
    )
    train_idx, test_idx = X_temiz_train.index, X_temiz_test.index
    X_hafif_train = df.loc[train_idx, "yorum_hafif"]
    X_hafif_test = df.loc[test_idx, "yorum_hafif"]

    # ---------------- Yaklaşım 1: TF-IDF + Logistic Regression ----------------
    print("\n[Yaklaşım 1] TF-IDF + Logistic Regression")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=3000, min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_temiz_train)
    X_test_tfidf = tfidf.transform(X_temiz_test)

    logreg_tfidf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    logreg_tfidf.fit(X_train_tfidf, y_train)

    f1_train_tfidf = f1_score(y_train, logreg_tfidf.predict(X_train_tfidf), pos_label="positive")
    f1_test_tfidf = f1_score(y_test, logreg_tfidf.predict(X_test_tfidf), pos_label="positive")
    print(f"  Train F1: {f1_train_tfidf:.3f}   Test F1: {f1_test_tfidf:.3f}")

    # ---------------- Yaklaşım 2: Türkçe Embedding + Logistic Regression ------
    print("\n[Yaklaşım 2] Türkçe BERT Embedding + Logistic Regression")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    t0 = time.time()
    X_train_emb = embed_model.encode(X_hafif_train.tolist(), show_progress_bar=False)
    X_test_emb = embed_model.encode(X_hafif_test.tolist(), show_progress_bar=False)
    print(f"  Embedding süresi: {time.time() - t0:.1f} sn")

    # C=0.01: overfitting'i azaltmak için güçlü regularization (bkz. RAPOR.md, bölüm 4)
    logreg_emb = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=0.01)
    logreg_emb.fit(X_train_emb, y_train)

    f1_train_emb = f1_score(y_train, logreg_emb.predict(X_train_emb), pos_label="positive")
    f1_test_emb = f1_score(y_test, logreg_emb.predict(X_test_emb), pos_label="positive")
    print(f"  Train F1: {f1_train_emb:.3f}   Test F1: {f1_test_emb:.3f}")

    # ---------------- Karşılaştırma ----------------
    print("\n" + "=" * 50)
    print("KARŞILAŞTIRMA (aynı test seti, f1-score)")
    print("=" * 50)
    print(f"{'Yaklaşım':<35}{'Test F1':>10}")
    print(f"{'TF-IDF + LogisticRegression':<35}{f1_test_tfidf:>10.3f}")
    print(f"{'Embedding + LogisticRegression':<35}{f1_test_emb:>10.3f}")

    print("\nDetaylı rapor — Embedding (final model):")
    print(classification_report(y_test, logreg_emb.predict(X_test_emb)))


if __name__ == "__main__":
    main()