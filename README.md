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

<img width="750" height="450" alt="{7F5DF5B6-F1B0-4A10-AF66-AAF1D1C19F94}" src="https://github.com/user-attachments/assets/df4f5ea8-5702-475a-80bf-bbb71470fa4c" />


1 - ) To Do:
<img width="350" height="350" alt="{55C69CFF-920B-4F51-8B49-99FAD6AA2FB7}" src="https://github.com/user-attachments/assets/66dfb74f-b760-4529-b672-805cc7627246" />

2 - ) Doing:
<img width="350" height="350" alt="{20E0BF8F-ED5D-419D-B4EA-AA194198FE7C}" src="https://github.com/user-attachments/assets/c4e42145-f991-4302-bf51-da056685d8f2" />

3 - ) Done:
<img width="350" height="350" alt="{57D4CCC7-5C73-421E-AE96-87257F3AA71D}" src="https://github.com/user-attachments/assets/87a24e13-2956-4a05-a2c9-b47ed11bed83" />
 
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

## 📸 2.8. Ekran Görüntüleri

<img width="1910" height="639" alt="Screenshot 2025-07-20 191112" src="https://github.com/user-attachments/assets/84ac14cf-3aeb-4519-ae27-77a2fddbb7cb" />




---


# 🧠 YZTA-BootcampGroup33 - Sprint 3 Dokümantasyonu

## 📌 Proje Adı: DuyguDurumAI

### 🎯 Amaç  
Bireylerin metin tabanlı ifadelerinden duygu durumlarını analiz ederek olası ruhsal problemlere (anksiyete, depresyon vb.) dair erken uyarı/takip sistemi geliştirmek.

---

🗓 **Sprint Tarihi:** 22 Temmuz – 29 Temmuz

👥 **Takım Üyeleri**

* Kardelen – Scrum Master
* Melissa – Developer
* Duygu – Developer
* Oğuz – Developer

---

## 📝 3.1. Sprint Notları

* TF-IDF ve GloVe vektörleştirme görevleri tamamlandı.
* GeminE embedding entegrasyonu başarıyla test edildi, ancak optimizasyon gereksinimi gözlemlendi.
* XGBoost modeline ek olarak, baseline karşılaştırması için Logistic Regression,LightGBM,Random Forest,Naive Bayes uygulandı.
* Eğitim ve test skorları karşılaştırıldı.
* Model çıktılarının kullanıcıya sunumu için ilk arayüz tasarımı başlatıldı (CLI/Notebook tabanlı).
* Kodlama sırasında yaşanan versiyon problemleri, sanal ortam güncellemesi ile çözüldü.

---

## 📊 3.2. Sprintte Tamamlanması Gereken Puanlar

| User Story                                           | Tahmin (SP) | Tamamlandı mı?         |
| ---------------------------------------------------- | ----------- | ---------------------- |
| TF-IDF vektörleştirme entegrasyonu                   | 3           | ✅                      |
| GloVe embedding ile model eğitimi                    | 5           | ✅                      |
| GeminE embedding analizi ve optimizasyonu            | 5           | ❌ (Kısmen test edildi) |
| Alternatif model karşılaştırması (LogReg vs XGBoost) | 3           | ✅                      |
| ROC-AUC ve diğer metriklerle değerlendirme           | 3           | ✅                      |
| Basit kullanıcı arayüzü (CLI/Notebook demo)          | 3           | ✅                      |

**Toplam Tahmin SP:** 22
**Tamamlanan SP:** 19
**Kalan SP:** 3

---

## 🔍 3.3. Sprint Puanlama Mantığı

* Daha teknik, model karşılaştırmalı ve analiz içeren görevler yüksek puanlı olarak (5 SP) değerlendirildi.
* Arayüz prototipi gibi uygulanabilir, sınırlı kapsamlı işler orta düzey puan aldı (3 SP).
* GeminE vektörleştirmesi ile ilgili çalışma, veri boyutu ve kaynak gereksinimi nedeniyle tamamlanamadı.

---

## 📅 3.4. Daily Scrum

| Tarih | Üye      | Dünü Ne Yaptım?                             | Bugün Ne Yapacağım?                    | Blokaj Var mı?     |
| ----- | -------- | ------------------------------------------- | -------------------------------------- | ------------------ |
| 22.07 | Kardelen | Sprint görevlerini küçük task'lara böldüm   | Trello board'u güncelleyeceğim         | ❌                  |
| 23.07 | Melissa  | TF-IDF kodunu tamamladım                    | GloVe vektörlerini entegre edeceğim    | ❌                  |
| 24.07 | Oğuz     | GeminE için ilk embed örnekleri oluşturuldu | Bellek sorunlarını çözmeye çalışacağım | ✅ GPU yetersizliği |
| 25.07 | Duygu    | ROC-AUC hesaplaması yaptı                   | Model metriklerini görselleştireceğim  | ❌                  |

---

## 📌 3.5. Sprint Board Güncellemeleri

<img width="750" height="450" alt="{C8D43BAE-2032-4CFA-9C85-249ED4C784BD}" src="https://github.com/user-attachments/assets/a9d2aad3-0a81-4495-ac98-fefed42c10f2" />

1 - ) To Do:
<img width="350" height="350" alt="{3604BAA0-A7C2-4F15-897F-EE1E829F6C5E}" src="https://github.com/user-attachments/assets/3ddd3bff-6409-473a-9c39-58e43ad86f0b" />

2 - ) Doing:
<img width="350" height="350" alt="{2B0F4A1B-16B6-43D4-BADA-05698F07130E}" src="https://github.com/user-attachments/assets/159dd775-decc-479f-a515-6f662f5fb249" />

3 - ) Done:
<img width="350" height="350" alt="{81D22314-36DC-4F0F-B42E-4E220972B874}" src="https://github.com/user-attachments/assets/84e0405a-18fe-4cfc-90db-3af4c9a8af04" />



---

## ✅ 3.6. Sprint Review

* TF-IDF ve GloVe başarıyla entegre edildi, model doğrulukları ve metrikleri elde edildi.
* Accuracy, F1 ve doğruluk sonuçları karşılaştırıldı, LightGBM en başarılı model konumunda.
* Arayüzde kullanıcı girişi alınıp model çıktısı döndürüldü.
* GeminE entegrasyonu test aşamasında, ancak büyük boyutlu embed'ler RAM/GPU sorunu oluşturdu.

---

<img width="857" height="425" alt="Figure 2025-08-03 184205" src="https://github.com/user-attachments/assets/87a3b245-2218-4df0-a50e-645758db9f03" />
Eğitilen modellerin performans olarak karşılaştırılması

## 🔄 3.7. Sprint Retrospective

**İyi Gidenler:**

* Model karşılaştırmaları ve metrik analizleri netlik kazandırdı.
* Teknik görevlerde iş birliği kuvvetliydi.
* Arayüz için somut bir adım atıldı.

**Geliştirilmesi Gerekenler:**

* Embedding’lerde kaynak problemi yaşandı (RAM/GPU).
* GloVe ile model başarıları tatmin edici ancak daha fazla analiz gerekebilir.

**Aksiyonlar:**

* Sprint 4'te GeminE optimizasyonu için colab GPU kullanımına geçilecek.
* SHAP ve LIME ile model açıklanabilirliği (explainability) artırılacak.
* Arayüz kullanıcı deneyimi için iyileştirilecek.

---

## 📸 3.8. Ekran Görüntüleri

<img width="350" height="500" alt="{9DF750F8-65CB-4543-8831-7E03F04C49B6}" src="https://github.com/user-attachments/assets/dc084e87-71bd-4421-abf0-47ea9deef099" />

<img width="500" height="500" alt="{A966FBD5-7F4C-4F6E-938F-CF29C6D80FCD}" src="https://github.com/user-attachments/assets/dd590506-952f-4fde-990e-85d8d363210f" />






---

