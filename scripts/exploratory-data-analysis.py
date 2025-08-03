# Gerekli kütüphanelerin içe aktarılması
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import hstack
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# Veriyi CSV dosyasından oku
df = pd.read_csv(raw_data)

# Verinin ilk 5 satırını göster
df.head()

# Sayısal sütunların temel istatistiklerini göster
df.describe()

# Veri kümesinin genel bilgilerini göster (sütun tipleri, eksik değerler vs.)
df.info()

# Eksik değerlerin sayısını sütun bazında göster
df.isna().sum()

# 'status' sütunundaki değerlerin dağılımını say
df.status.value_counts()

# 'status' değerlerinin dağılımını bir değişkende tut
status_counts = df['status'].value_counts()

#renk paleti
colors = ['#419D78', '#E0A458', '#2D3047', '#FFDBB5', '#C04ABC', '#B3CDE0', '#D0D0D0']

# Pasta grafiği çizmek için figür boyutunu ayarla
plt.figure(figsize=(7, 7))

# 'status' dağılımını pasta grafiği olarak çiz
plt.pie(
    status_counts,
    labels=status_counts.index,
    autopct='%1.1f%%',    # yüzde formatı
    startangle=140,       # grafiğin başlangıç açısı
    colors=colors,        # renk paleti (önceden tanımlanmış olmalı)
    shadow=True           # gölge efekti
)

# Grafiğe başlık ekle
plt.title('Distribution of Mental Health Conditions')

# Pasta grafiğini daire şeklinde göster
plt.axis('equal')

# Yerleşimi düzenle ve grafiği göster
plt.tight_layout()
plt.show()

# Her bir 'status' için rastgele birer 'statement' seç
random_statements = df.groupby('status').sample(n=1)['statement']

# Rastgele seçilen ifadeleri ekrana yazdır
for status, statement in random_statements.items():
    print(f"Status: {status}")
    print(f"Statement: {statement}\n")

# Her bir 'statement' için karakter sayısını hesapla ve yeni bir sütuna ekle
df['num_of_characters'] = df['statement'].str.len()

# Her bir 'statement' için cümle sayısını hesapla ve yeni bir sütuna ekle
df['num_of_sentences'] = df['statement'].apply(lambda x: len(nltk.sent_tokenize(x)))

# Karakter ve cümle sayısı sütunlarının istatistiksel özetini al
description = df[['num_of_characters', 'num_of_sentences']].describe()

# Karakter sayısı 10,000'den büyük olan satırları filtrele
df[df['num_of_characters'] > 10000]
