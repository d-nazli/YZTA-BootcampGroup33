# 🧠 YZTA-BootcampGroup33 - Sprint 1 Dokümantasyonu

## 📌 Proje Adı: DuyguDurumAI

### 🎯 Amaç  
Bireylerin metin tabanlı ifadelerinden duygu durumlarını analiz ederek olası ruhsal problemlere (anksiyete, depresyon vb.) dair erken uyarı/takip sistemi geliştirmek.

### 🗓 Sprint Tarihi: 1 Temmuz – 5 Temmuz

### 👥 Takım Üyeleri
- **Kardelen** – Scrum Master  
- **Mesut** – Product Owner  
- **Melissa** – Developer  
- **Duygu** – Developer  
- **Oğuz** – Developer  

---

## 📝 1. Sprint Notları

- Projenin ana problemi, hedef kullanıcı kitlesi ve kullanım senaryoları belirlendi.  
- Veri kaynakları araştırılıyor. (Twitter, Reddit, Kaggle veri setleri, Emotion Dataset, DailyDialog)  
- Model tipi olarak metin sınıflandırmaya dayalı RNN/CNN ve transformer mimarileri değerlendiriliyor.  
- Projeye uygun Python kütüphaneleri ve mimari yapılar hakkında bilgi paylaşımı yapıldı.  
- Görev dağılımları belirlendi, her üyenin aktif kod katkısı sağlayacağı şekilde planlama yapıldı.  

---

## 📊 1.2. Sprintte Tamamlanması Gereken Puanlar

| User Story                                            | Tahmin (SP) | Tamamlandı mı? |
| ----------------------------------------------------- | ----------- | -------------- |
| Proje probleminin tanımı ve kullanım senaryoları      | 3           | ✅              |
| Etik değerlendirme ve kullanıcı mahremiyet stratejisi | 1           | ✅              |
| Veri seti araştırması                                 | 5           | ✅              |

---

## 🔍 1.3. Sprint Puanlama Mantığı

- **1 SP:** Basit görev / 1 saatten kısa süren işler  
- **3 SP:** Orta seviye / araştırma ve kodlama içeren işler  
- **5 SP:** Karmaşık, çok adımlı, model tasarımı veya analiz gibi görevler  
- Story point’ler ekip kararı ile belirlendi.

---

## 📅 1.4. Daily Scrum

| Tarih  | Üye      | Dünü Ne Yaptım?                     | Bugün Ne Yapacağım?                        | Blokaj Var mı?        |
|--------|----------|-------------------------------------|--------------------------------------------|------------------------|
| 01.07  | Kardelen | Sprint planını hazırladım           | Görev dağılımını finalize edeceğim         | ❌                     |
| 02.07  | Melissa   | Kaggle veri setini inceledim        | Tokenizasyon kodlarını yazacağım           | ❌                     |
| 03.07  | Oğuz     | EDA planlaması yapılıyor            | PCA ile görselleştirme deneyeceğim         | ✅ Pandas sürüm sorunu |
| 04.07  | Duygu    | Etiket kategorileri araştırılıyor   | Clean text fonksiyonlarını yazacağım       | ❌                     |
| 05.07  | Mesut    | Kullanıcı akışını çizdim            | MVP için gereksinim listesini çıkartacağım | ❌                     |

---

## 📌 1.5. Sprint Board Güncellemeleri

**To Do:**
- Modelin seçimi

**In Progress:**
- Keşifsel Veri Analizi  
- Duygu sınıflarının araştırılması  
- Veri seti araştırması  

**Done:**
- Ekip içi rol dağılımı  
- Proje fikri belirlenmesi  
- Kullanım senaryosu belirleme  

---

## ✅ 1.6. Sprint Review

- Sprint 1’de hedeflenen tüm görevler başarıyla tamamlandı.  
- Takım iş birliği çok yüksekti, tüm üyeler projeye katkı verdi.  
- Kullanıcı odaklı senaryolar üretildi ve minimum ürün için çerçeve çizildi.  

---

## 🔄 1.7. Sprint Retrospective

**İyi Gidenler:**
- Herkes aktif şekilde katkı verdi.  
- Görev dağılımı adil ve verimliydi.  
- Daily scrum’lar düzenliydi.  

**Geliştirilmesi Gerekenler:**
- Ortak klasör düzeni başta karmaşayla ilerledi.  
- Kod standardizasyonu eksikliği hissedildi.  

**Aksiyonlar:**
- Ortak bir repo yapısı oluşturulacak.  
- Sprint 2 başında “kod stili ve klasör yapısı rehberi” yayınlanacak.  
- Mini teknik bilgi sunumları yapılacak (örneğin: Melisa’nın tokenizer eğitimi, Oğuz’dan matplotlib tips).  

---

# 🧠 YZTA-BootcampGroup33 - Sprint 2 Dokümantasyonu

## 📌 Proje Adı: DuyguDurumAI

### 🎯 Amaç  
Bireylerin metin tabanlı ifadelerinden duygu durumlarını analiz ederek olası ruhsal problemlere (anksiyete, depresyon vb.) dair erken uyarı/takip sistemi geliştirmek.

### 2 Sprint Tarihi: 7 Temmuz – 20 Temmuz

### 👥 Takım Üyeleri
- **Kardelen** – Scrum Master  
- **Mesut** – Product Owner  
- **Melissa** – Developer  
- **Duygu** – Developer  
- **Oğuz** – Developer  

---

## 📝 2.1. Sprint Notları

- Ön işleme (preprocessing) adımları geliştirildi: stopword removal, küçük harfe çevirme, noktalama işaretlerinin kaldırılması tamamlandı.  
- XGBoost modeli başarıyla eğitildi ve ilk doğruluk sonuçları elde edildi.  
- TF-IDF ve diğer vektörleştirme yöntemleri (GloVe, GeminE benzeri embedding) planlandığı hâlde tamamlanamadı.  
- Sprint planına kıyasla ilerleme eksik kaldı, ancak temel model kurulumları tamamlandığı için temel yapı sağlamlaştı.

---

## 📊 2.2. Sprintte Tamamlanması Gereken Puanlar

| User Story                                               | Tahmin (SP) | Tamamlandı mı? |
|-----------------------------------------------------------|-------------|----------------|
| Metin verisinin temizlenmesi ve ön işlenmesi              | 5           | ✅              |
| TF-IDF vektörleştirme uygulaması                          | 3           | ❌              |
| XGBoost modelinin eğitilmesi                              | 5           | ✅              |
| GloVe / GeminE vektörleştirme yöntemlerinin entegrasyonu  | 5           | ❌              |
| Eğitim doğruluğunun ölçülmesi ve raporlanması             | 3           | ✅              |

**Toplam Tahmin SP:** 21  
**Tamamlanan SP:** 13  
**Kalan SP:** 8  

---

## 🔍 2.3. Sprint Puanlama Mantığı

- Görevler karmaşıklık ve tahmini süreye göre 1–5 arası puanlandı.  
- Vektörleştirme işlemleri orta düzeyde (3 SP), model eğitimi ve embedding entegrasyonu yüksek puanlı (5 SP) olarak değerlendirildi.  
- Sprint sonunda kalan görevler, Sprint 3’e aktarılacak.

---

## 📅 2.4. Daily Scrum

| Tarih  | Üye      | Dünü Ne Yaptım?                            | Bugün Ne Yapacağım?                                 | Blokaj Var mı?           |
|--------|----------|---------------------------------------------|------------------------------------------------------|--------------------------|
| 06.07  | Kardelen | Sprint backlog'u oluşturduk                | Story point'leri güncelleyeceğim                     | ❌                        |
| 07.07  | Melissa   | Temizleme fonksiyonlarını yazdım            | TF-IDF üzerine çalışacağım                           | ✅ Zaman yetmedi          |
| 08.07  | Duygu    | XGBoost modeli kurdu                        | Doğruluk skorlarını analiz edeceğim                  | ❌                        |
| 09.07  | Oğuz     | Normalizasyon kodlarını yazdı               | GeminE için araştırma yapacağım                      | ✅ Kaynak eksikliği       |
| 10.07  | Mesut    | Kullanıcı için çıktı formatları planlandı   | TF-IDF analiz sonuçlarını özetleyeceğim              | ❌                        |

---

## 📌 2.5. Sprint Board Güncellemeleri

**To Do:**
- GeminE / GloVe entegrasyonu  
- TF-IDF grafiksel analizi  
- XGBoost çıktı optimizasyonu  

**In Progress:**
- TF-IDF kodları  
- Model sonuçlarının yorumlanması  

**Done:**
- Temiz metin yapısı oluşturuldu  
- XGBoost modeli eğitildi  
- Doğruluk ve hata metrikleri hesaplandı  

---

## ✅ 2.6. Sprint Review

- Sprint 2’de önemli teknik görevler (temizleme ve model eğitimi) başarıyla tamamlandı.  
- TF-IDF ve embedding çalışmaları zaman ve kaynak kısıtları nedeniyle Sprint 3’e devredildi.  
- Mevcut model, baseline sonuçlar için hazır hâle getirildi.  
- Ekibin teknik katkısı devam etti ancak yoğunluk nedeniyle bazı görevlerde gecikme yaşandı.

---

## 🔄 2.7. Sprint Retrospective

**İyi Gidenler:**
- XGBoost modeli başarıyla çalıştırıldı ve ilk doğruluk sonuçları üretildi.  
- Takım üyeleri birbirinin kodlarını test etti, iş birliği kuvvetliydi.  

**Geliştirilmesi Gerekenler:**
- Zaman yönetimi Sprint 2’de zorlandı.  
- Vektörleştirme gibi araştırma içeren görevlerde teknik destek ihtiyacı hissedildi.  

**Aksiyonlar:**
- Sprint 3 başında vektörleştirme görevleri önceliklendirilecek.  
- Melissa ve Oğuz embedding üzerine birlikte çalışacak.  
- Kardelen Sprint 3 için daha kısa task'lar halinde planlama yapacak.

## 2.8. Ekran Görüntüleri

<img width="1910" height="639" alt="Screenshot 2025-07-20 191112" src="https://github.com/user-attachments/assets/84ac14cf-3aeb-4519-ae27-77a2fddbb7cb" />
<img width="637" height="712" alt="Screenshot 2025-07-20 194531" src="https://github.com/user-attachments/assets/99201f71-42ac-4dd6-879f-4c3bd82ff35d" />
<img width="639" height="755" alt="Screenshot 2025-07-20 194630" src="https://github.com/user-attachments/assets/462c6891-034f-471c-9223-4d71aec4a234" />
<img width="638" height="751" alt="Screenshot 2025-07-20 194737" src="https://github.com/user-attachments/assets/55f47ab5-6a4a-48d5-8f53-648164087f1b" />



---

