# Veri Sözlüğü — `tr_yorumlar.csv`

Gerçek Türkçe ürün yorumları. **~570 satır** (her satır bir yorum), **2 kolon**. Veri **dengeli** — yaklaşık yarısı pozitif, yarısı negatif.

> Bu **ham** bir metin verisidir; gerçek insanların yazdığı yorumlardır. Yazım hataları, kısaltmalar, konuşma diline özgü ifadeler içerir.

## Kolonlar

| Kolon | Açıklama | Beklenen değerler |
|---|---|---|
| `yorum` | Müşterinin yazdığı ham yorum metni | serbest metin (Türkçe) |
| `duygu` | **Hedef değişken** — yorumun duygu etiketi | `positive`, `negative` |

## Notlar

- **Hedef değişken** `duygu`. Amacın bunu tahmin etmek.
- Metin ham haldedir — büyük/küçük harf, noktalama, rakam vb. senin işlemen gerekebilir. Türkçe'nin küçük harfe çevirme davranışının İngilizce'den farklı olduğunu unutma.
- Veri dengeli olduğu için sınıf dağılımı bir sorun değil; ama bu, metrik seçimini yine de düşünmen gerekmediği anlamına gelmez.
