# Yorum Duygu Analizi — Rapor

> Bu şablonu doldur ve dosyayı `RAPOR.md` olarak kaydet. Teknik olmayan bir yöneticinin
> anlayacağı **iş diliyle** yaz. Kısa ve net olması yeterli — uzunluk değil, netlik önemli.
> Başlıkları silme; altlarını kendi cümlelerinle doldur.

## 1. Özet (3-4 cümle)
Ne yaptın, hangi yöntemleri denedin, hangisi kazandı, sonuç ne?
-Data Sınıflandırmak için iki istenildiği üzere iki farklı yaklasım denedim.
-TF-IDF ve LogisticRegresyon diğeride Türkçe özel Bert tabanlı embeding modeli kullandım .
-Her ikisinide aynı train/test le eğitildi ve f1 scorlar bakıldı 
-Embedding olan daha iyi sonuc verdi buda çıktı (Test F1 = 0.879 vs 0.813)


## 2. Metni hazırlama (ön işleme)
Türkçe metinde hangi sorunları/tuzakları fark ettin? Her birini **nasıl** ve **neden** öyle çözdün?
-Metin de genel anlamda bir sorun vardı zıtlık bağlaçları kelimeler bazen olmlu olurken bazen olumsuz olabiliyordu .
-Standart lower yani küçültme hata verdi buda sorun çıkardı 
-Düzgün yazılmayan yorumlar yani yazım hatası gibi veriler sorun çıkardı 
-Tekrarlayan harfler sorun çıkardı 





## 3. Denediğin yaklaşımlar ve karşılaştırma
En az iki yaklaşımı anlat. Nasıl adil karşılaştırdın (aynı ayrım, aynı test)? Sonuçları
sayılarla (f1) yan yana koy. Hangisini önerirsin?
-TF-IDF + Logistic Regression kullandım bunun sonucu test f1 0,813 çıktı
-Bert Embeding + LogisticRegresion kullandım sonucu test f1 0,879 çıktı 
-embeding yaklasımı daha iyi bir sonuc verdi 



## 4. Neden bu sonuç çıktı?
Kazanan yöntem neden kazandı / kaybeden neden kaybetti? (En gelişmiş yöntem her zaman
kazanmaz — buradaki durumu yorumla.)
-Embeding daha iyi bir sonuc verdi ancak ilk denemelerde f1 scora 1'e kadar çıktı buda yapının ezberlmesini sağladı. 
-Embeding kalıplaşmış olumsuzlukları daha iyi tespit ediyor ancak yanıldığı yerlerde var (ama/fakat) gibi ifadelerde yanılıyor. 




## 5. Model nerede yanılıyor? (hata analizi)
Yanlış sınıflanan yorumlara baktığında ortak bir örüntü var mı (ironi, çok kısa/bağlamsız
yorumlar, hem övgü hem şikayet)? Örnek ver.
-Embeding küfür içeren cümleler ve yapıda hata alıyor bilemiyor .
Test setinde 114 yoruma baktım 84 yorumda iki model de doğru bildi
16 yorumda sadece TF-IDF yanıldı , 7 yorumda sadece embedding yanıldı 



## 6. Tavsiyen
Bu analizle ürün ekibine **ne yapmasını** önerirsin? Somut bir aksiyon yaz.
-Türkçe Embeding modeli kullanmamız gerek ancak buda iyileştirmeler yapıldıktan sonra 
örneğin zıtlık içeren yorumlarda daha maneul bir yapı olabilir 
-Daha temiz bir data da tekrar eğitimi olabilir . 


## 7. (Opsiyonel) Daha fazla zamanın olsa
Neyi denerdin / neyi geliştirirdin?
-Yazım hatalarını otomatik düzeltecek fonskiyon eklenebilir 
-Kategoriler eklenebilir 
-Ters zıtlıkta büyük hata almsıtın onun için daha iyi eğiteccek bir data olabilir .
