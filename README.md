Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
yapılan diyabet araştırması için kullanılan verilerdir.
![image](https://user-images.githubusercontent.com/84075546/149348854-78b37405-4bdd-4b62-a57c-46ee300daf24.png)



-Pregnancies              :Hamilelik sayısı
-Glucose                  :Oral glikoz tolerans testinde 2 saatlik 
                           plazma glikoz konsantrasyonu
-BloodPressure            :Kan Basıncı (Küçük tansiyon) (mm Hg)
-SkinThickness            :Cilt Kalınlığı
-Insulin                  :2 saatlik serum insülini (mu U/ml)
-BMI                      :Vücut kitle endeksi
-DiabetesPedigreeFunction :Fonksiyon (Oral glikoz tolerans testinde 2 
                           saatlik plazma glikoz konsantrasyonu)
-Age                      :Yaş (yıl)
-Outcome                  :Hastalığa sahip (1) ya da değil (0)
Diyabet çeşitlerinin ayrıntısına girmeden verilen değişkenlerimiz ile diyabet ilişkisini gözden geçirelim. 

Diyabet çeşidi 2 ye ayrılır:

İstisnalar olmakla birlikte Tip 1 diyabet hastaları daha genç ve zayıf olma eğilimindeyken, Tip 2 diyabet hastaları daha ileri yaşlarda ve kilolu kişilerdir

Pankreas insulin hormonu üretemez→kanda şeker artışı→idrara gidiş artışı→glikoz atılması artar→kalori atımı artar→kilo kaybı yaşanır→Tip1 diyab





# diabet.predct

# Keşifçi Veri Analizi
# Adım 1: Genel resmi inceleyiniz.
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
# Adım 5: Aykırı gözlem analizi yapınız.
# Adım 6: Eksik gözlem analizi yapınız.
# Adım 7: Korelasyon analizi yapınız.
#######   Görev 2 :
#
# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.
# Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
# değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

# Adım 2: Yeni değişkenler oluşturunuz.
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
# Adım 5: Model oluşturunuz.
