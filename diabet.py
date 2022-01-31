# Görev 1 :

# Keşifçi Veri Analizi
# Adım 1: Genel resmi inceleyiniz.
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
# Adım 4: Hedef değişken analizi yapınız.
# (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre numerik değişkenlerin ortalaması)
# Adım 5: Aykırı gözlem analizi yapınız.
# Adım 6: Eksik gözlem analizi yapınız.
# Adım 7: Korelasyon analizi yapınız.
# Görev 2 :

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

# Problem: Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
# edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir. Modeli
# geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği adımlarını
# gerçekleştirmeniz beklenmektedir.

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
# Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
# yapılan diyabet araştırması için kullanılan verilerdir.
# Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir




import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Veri setimizi yükleyip ilk 5 gözleme bakalım ve değişkenlerimizi tanıyalım:
def load():
    data=pd.read_csv("hafta6/feature_engineering/diabetes.csv")
    return data
df = load()
df.head(10)

# Veri setimizde gözlem ve değişken sayısımıza bakalım:
df.shape # (768, 9) satır-sütun sayısı
df.describe().T # bazı değişkenlerin (Insulin,Glucose,BloodPressure,BMI,SkinThickness) min değeri "0",olmaması gereken bir durum
df.median().T

# aykırı değerlere kutu grafikleri ile bakalım
sns.boxplot(x=df["Insulin"])
plt.show()

# Sayısal değişkenleri,kategorik değişkenleri,kategorik olmasa bile aslında kategorik olan,
# kategorik olduğu halde aslında kardinal olan değişkenleri kontrol edelim
df.info()
# Veri setimiz sayısal değişkenlerden oluşmakta, fonksiyon oluşturup kontrol edelim.
# Ayrıca non_null olduğunu söylemiş fakat boş değil gibi gözükse de insulin ve glikoz
# değerleri sıfır olamayacağı halde sıfır yazılmış bunlar ile ilgili de işlem yapalım:
def grab_col_names(dataframe, cat_th=5, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols
cat_but_car

# Değişkenlerin toplamlarını alarak groupby ile ilişki kurabilcek miyiz bakalım:
df.groupby("Outcome")[["Pregnancies","Glucose","BloodPressure","DiabetesPedigreeFunction","BMI","Insulin"]].agg("sum")
# Hasta olmayanların değerleri daha fazla gibi bu çıktı bize yardımcı olmuyor.

# groupby ı bir de ortalama ve median alarak yapalım:
df.groupby("Outcome")[["Pregnancies","Glucose","BloodPressure","DiabetesPedigreeFunction","BMI","Insulin"]].agg("mean")
df.groupby("Outcome")[["Pregnancies","Glucose","BloodPressure","DiabetesPedigreeFunction","BMI","Insulin"]].agg("median")

# num_cols ta bulunan sayısal değişkenlerin ortalamalarına bakmak isteyelim:
df.loc[:,num_cols].apply(lambda x: x.mean()).head()

# num_cols listesinin içerisindeki değişkenlere ulaşıp diyabet hastası olan ve olmayanların
# ortalamasını fonksıyon oluşturup almak isteyelim:
# aykırı değerlerin fazla olması durumunda ortalama değil median a göre işlem daha doğru yapmak olduğundan
# bir de median alalım:
for i in num_cols:
    print(df.groupby("Outcome")[i].mean())

for i in num_cols:
    print(df.groupby("Outcome")[i].median())


# Aradaki farka dikkat edersek Insulin,Glikoz,BMI,Age değerlerinde fark daha da arttığı gözlemlenir.

# bu gösterimi hedef değişkene göre farklı bir fonksıyon ıle de gösterılebılır( Görseli daha güzel)
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Outcome", col)

# Diyabet hastalığını değişkenler ile ilişkilerini yorumlamadan daha,
# aykırı gözlem analizi,eksik gözlem analizi yapalım:
sns.boxplot(x=df["Age"])
plt.show()
sns.boxplot(x=df["BloodPressure"])
plt.show()
sns.boxplot(x=df["Glucose"])
plt.show()
sns.boxplot(x=df["DiabetesPedigreeFunction"])
plt.show()
sns.boxplot(x=df["BMI"])
plt.show()
sns.boxplot(x=df["Insulin"])
plt.show()
sns.boxplot(x=df["Pregnancies"])
plt.show()
sns.boxplot(x=df["SkinThickness"])
plt.show()
# 9 değişkenden sıfır değeri alamıyacak olanlar: Insulin,Glucose,BloodPressure,BMI,SkinThickness
# alabılır olanlar: age(zaten 21 den başlıyor),hamılelık,outcome,DiabetesPedigreeFunction
without_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[without_zero] = df[without_zero].replace(['0', 0], np.nan) # 0 ları "NAN" ile doldurduk

df.head()
df.isnull().sum() # eksik değerlerin toplamlarına bakılır
df = df.apply(lambda x: x.fillna(x.median()) if x.dtype!="O" else x ,axis=0) # eksik değerler median ile dolduruldu
df.head()

# The dark color shows the high correlation between the variables and the light
# colors shows less correlation between the variables
# Bu corr analizine göre Pregnancy-BMI,Insulin,Glucose ile yüksek ilişkili
corr=df.corr()
corr.style.background_gradient(cmap='coolwarm')
sns.heatmap(df.corr())

# numerik değişkenlerin histogramlarına bakılmak istenirse:
plt.savefig("Plotting_Correlation_HeatMap.jpg")
df.hist(bins=50, figsize=(20,10))

# Aykırı değerleri belirlediğimiz aralığa göre yakalayıp üst limitleri üst limitle
# alt limitleri alt limitle dolduralım
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
check_outlier(df,"Age")
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
df.head()
grab_outliers(df, "Pregnancies")
grab_outliers(df, "Age")
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit # alt limitten aşağıda olan limiti alt limitle
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


check_outlier(df,"Age")

# sütunlarda dolaşıp min ve max değerlerine bakalım
for i in df.columns:
    print( i + " -> " + "max value is " + ": " + str(df[i].max()) + " and " + "min value is " + ": " + str(df[i].min()))

# yeni değişkenler üretelim:

df["age_desc"] = pd.cut(x=df["Age"], bins=[20,30,55,81], labels=["young","middle_aged","old"])
# Normal (BMI <25), kilolu (BMI=25-29,9) ve obez (BMI>29,9) olarak 3 guruba ayırdık.
# http://cms.galenos.com.tr/Uploads/Article_21227/IMJ-14-243-En.pdf

df["BMI_desc"] = pd.cut(x=df["BMI"],bins=[17,24,30,70],labels=["normal","overweight","obese"])
df.head()
# https://jag.journalagent.com/terh/pdfs/TERH_24_1_37_42.pdf

# A blood sugar level less than 140 mg/dL (7.8 mmol/L) is normal. A reading of more than 200 mg/dL (11.1 mmol/L)
# after two hours indicates diabetes. A reading between 140 and 199 mg/dL (7.8 mmol/L and 11.0 mmol/L) indicates
# prediabetes.
# https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451

df["glucose_desc"] = pd.cut(x=df["Glucose"],bins=[40,139,198,230],
                            labels=["normal","prediabetes","diabet"])

df["BloodPressure_desc"] = pd.cut(x=df["BloodPressure"],bins=[str(df["BloodPressure"].min()-1),79,89,99,109,str(df["BloodPressure"].max())],
                                  labels=["normal","prehypertension","hypertension_stage1","hypertension_stage2","hypertensive_crisis"])
df.head()

df['Insulin/SkinThickness']=df['Insulin']/df['SkinThickness']
df.groupby('Outcome')['Insulin/SkinThickness'].mean()
df.groupby('Pregnancies')['Insulin/SkinThickness'].mean()

df['DiabetesPedigreeFunction/BloodPressure']=df['DiabetesPedigreeFunction']/df['BloodPressure']
df['Insulin/BloodPressure']=df['Insulin']/df['BloodPressure']

# yeni değişkenler ile cat,num,car sayıları değişti kontrol edelim:
cat_cols,num_cols,cat_but_car=grab_col_names(df)

# yeni üretilen değişkenleri one hote encoder ile sütunlara verildi ve o satırda gözlemleniyorsa 1,
# gözlemlenmıyorsa 0 denildi:
# 2 sınıftan oluşuyorsa bınary col denir. "Outcome" değişkenımız 0-1 lerden zaten oluşuyor,diğerlerini encode edersek:

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]
cat_cols
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

# görselleştirme
plt.figure(figsize=(15,15))
plt.title('Per Age Pregnancies')
sns.lineplot(x="Age", y="Age", hue="Pregnancies" ,data=df);
# 30-60 yaş arası hamilelik sayısı daha fazla

# catsummary ıle kategorık degıskenlerın kendı ıcınde dagılımları gorduk
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe) # frekansları görselleştırmek istersek plot= true
        plt.show()

for col in cat_cols:
    cat_summary(df, col,True)







from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
# algoritma perfonmansı yükseltmesi açısından değerler standartlaştırma ile küçültülür:
num_cols
df[num_cols]=scaler.fit_transform(df[num_cols])
df.head()
df.isnull().sum() # eksik yok

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)   # Out[202]: 0.7922077922077922

# oluşturulan model sonucu : girilen değerler ile % 79 diyabet hastası olıup olmadığı tahmin edilebilir denir

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=16)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# model oluşturup hasta olup olmamasını tahmın gucumuzu bulalım
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)    # 17 yapınca 42 yı 0.78 çıktı

from sklearn.ensemble import RandomForestClassifier
df.head()
df.isnull().sum()
df.info()
df.isnull().values.any()
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# yeni değişkenler

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(20,10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
