import pytest
import sys
import os
import numpy as np
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tasks.task_manager import (
    load_data, explore_data, clean_text, preprocess, split_data,
    train_tfidf, train_lsa, evaluate, compare_methods, error_analysis,
)


# ──────────────────────────────────────────────────────
# Modül-seviye cache — testler arası tekrar eğitme yok
# ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def df():
    return load_data()


@pytest.fixture(scope="module")
def split(df):
    return split_data(df)


@pytest.fixture(scope="module")
def tfidf_model(split):
    X_train, X_test, y_train, y_test = split
    return train_tfidf(X_train, y_train)


@pytest.fixture(scope="module")
def lsa_model(split):
    X_train, X_test, y_train, y_test = split
    return train_lsa(X_train, y_train)


# 1. load_data
def test_load_data_structure(df):
    assert isinstance(df, pd.DataFrame)
    assert 'yorum' in df.columns and 'duygu' in df.columns
    # Gerçek/büyük veri seti — küçük kurgusal değil
    assert len(df) > 400
    assert set(df['duygu'].unique()) <= {'positive', 'negative'}


# 2. explore_data
def test_explore_data(df):
    stats = explore_data(df)
    assert set(stats.keys()) == {'n_rows', 'n_positive', 'n_negative', 'avg_length'}
    assert stats['n_rows'] == len(df)
    # Dengeli veri seti — pozitif/negatif birbirine yakın olmalı
    total = stats['n_positive'] + stats['n_negative']
    assert total == stats['n_rows']
    ratio = min(stats['n_positive'], stats['n_negative']) / max(stats['n_positive'], stats['n_negative'])
    assert ratio > 0.85
    assert stats['avg_length'] > 0


# 3. clean_text
def test_clean_text_basic():
    out = clean_text("Bu ÜRÜN çok İYİ!")
    assert out == out.lower()
    assert '!' not in out
    # Türkçe'ye duyarlı küçük harf: İYİ -> iyi (İ -> i)
    assert 'iyi' in out
    assert 'ürün' in out


def test_clean_text_removes_punctuation_and_digits():
    out = clean_text("Kargo 5 günde geldi, çok hızlı!!! 123")
    assert ',' not in out
    assert '!' not in out
    assert not any(ch.isdigit() for ch in out)
    assert 'çok' in out
    assert 'hızlı' in out


def test_clean_text_collapses_whitespace():
    out = clean_text("çok   fazla     boşluk")
    assert '  ' not in out
    assert out == out.strip()


# 4. preprocess
def test_preprocess_series(df):
    cleaned = preprocess(df['yorum'].head(20))
    assert isinstance(cleaned, pd.Series)
    assert len(cleaned) == 20
    for text in cleaned:
        assert text == text.lower()
        assert not any(ch.isdigit() for ch in text)


# 5. split_data
def test_split_data_shapes(df, split):
    X_train, X_test, y_train, y_test = split
    total = len(df)
    # %80/%20 ayrım (küçük yuvarlama farkı tolere edilir)
    assert abs(len(X_test) / total - 0.2) < 0.02
    assert len(X_train) + len(X_test) == total
    assert set(pd.unique(y_train)) <= {0, 1}
    assert set(pd.unique(y_test)) <= {0, 1}


def test_split_data_stratified(df, split):
    X_train, X_test, y_train, y_test = split
    full_ratio = df['duygu'].eq('positive').mean()
    test_ratio = y_test.mean()
    # Stratify sayesinde oranlar yakın olmalı
    assert abs(full_ratio - test_ratio) < 0.05


# 6. train_tfidf
def test_train_tfidf_predicts(split, tfidf_model):
    X_train, X_test, y_train, y_test = split
    preds = tfidf_model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds) <= {0, 1}


# 7. train_lsa
def test_train_lsa_predicts(split, lsa_model):
    X_train, X_test, y_train, y_test = split
    preds = lsa_model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds) <= {0, 1}


# 8. evaluate
def test_evaluate_metrics_range(split, tfidf_model):
    X_train, X_test, y_train, y_test = split
    metrics = evaluate(tfidf_model, X_test, y_test)
    assert set(metrics.keys()) == {'accuracy', 'precision', 'recall', 'f1'}
    for v in metrics.values():
        assert 0 <= v <= 1


def test_evaluate_tfidf_f1_threshold(split, tfidf_model):
    X_train, X_test, y_train, y_test = split
    metrics = evaluate(tfidf_model, X_test, y_test)
    # Türkçe sentiment'te TF-IDF (seyrek temsil) yüksek f1 vermeli
    assert metrics['f1'] >= 0.80


# 9. compare_methods
def test_compare_methods_structure(split, tfidf_model, lsa_model):
    X_train, X_test, y_train, y_test = split
    result = compare_methods(tfidf_model, lsa_model, X_test, y_test)
    assert set(result.keys()) == {'tfidf', 'lsa', 'better_f1'}
    assert set(result['tfidf'].keys()) == {'accuracy', 'precision', 'recall', 'f1'}
    assert set(result['lsa'].keys()) == {'accuracy', 'precision', 'recall', 'f1'}
    assert result['better_f1'] in {'tfidf', 'lsa'}


# 10. error_analysis
def test_error_analysis_structure(split, tfidf_model):
    X_train, X_test, y_train, y_test = split
    errors = error_analysis(tfidf_model, X_test, y_test, n=10)
    assert isinstance(errors, list)
    assert len(errors) <= 10
    for e in errors:
        assert set(e.keys()) == {'yorum', 'gercek', 'tahmin'}
        assert e['gercek'] != e['tahmin']
        assert e['gercek'] in (0, 1)
        assert e['tahmin'] in (0, 1)


def test_error_analysis_respects_n(split, tfidf_model):
    X_train, X_test, y_train, y_test = split
    errors = error_analysis(tfidf_model, X_test, y_test, n=3)
    assert len(errors) <= 3


# ──────────────────────────────────────────────────────
# Kaizu skor gönderimi — bu kısma DOKUNMA
# ──────────────────────────────────────────────────────

import requests


def _send_score(user_score):
    """Kaizu API'sine skor gönder. user_id ve project_id kaizu_config'ten gelir."""
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    try:
        from kaizu_config import USER_ID, PROJECT_ID
    except ImportError:
        print("⚠️  kaizu_config.py bulunamadı — skor gönderilmeyecek.")
        return

    if USER_ID == 0:
        print("⚠️  kaizu_config.py'de USER_ID=0 — kendi ID'ni yazmadın, skor gönderilmeyecek.")
        return

    url = "https://kaizu-api-8cd10af40cb3.herokuapp.com/projectLog"
    payload = {
        "user_id": USER_ID,
        "project_id": PROJECT_ID,
        "user_score": user_score,
        "is_auto": True,
    }
    try:
        r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        if r.status_code in (200, 201):
            print(f"✅ Skor gönderildi: {user_score}")
        else:
            print(f"⚠️  Skor gönderilemedi (HTTP {r.status_code})")
    except Exception as e:
        print(f"⚠️  Skor gönderilirken hata: {e}")


class _ResultCollector:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def pytest_runtest_logreport(self, report):
        if report.when == "call":
            if report.passed:
                self.passed += 1
            elif report.failed:
                self.failed += 1


def run_tests():
    """Tüm testleri çalıştır + skoru Kaizu'ya gönder."""
    collector = _ResultCollector()
    pytest.main([os.path.dirname(__file__), "-q"], plugins=[collector])
    total = collector.passed + collector.failed
    if total == 0:
        print("Hiç test çalışmadı.")
        return
    user_score = round((collector.passed / total) * 100, 2)
    print(f"\n📊 Toplam başarılı : {collector.passed}/{total}")
    print(f"📊 Skor            : {user_score}")
    _send_score(user_score)


if __name__ == "__main__":
    run_tests()
