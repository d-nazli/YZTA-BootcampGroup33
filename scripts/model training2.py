# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 12:28:50 2025

@author: SOGUTPC
"""
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             mean_squared_error, mean_absolute_error,
                             r2_score, precision_score, recall_score, f1_score)
import matplotlib.pyplot as plt
import numpy as np

# Model için özniteliklerin ayrılması
X = df[['tokens_stemmed', 'num_of_characters', 'num_of_sentences']]
y = df['status']

# Etiket kodlama ve train-test bölme
lbl = LabelEncoder()
y = lbl.fit_transform(y.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
#Sınıfların duygudurumlarla eşleştirilmesi
le_classes = np.array([0,1,2,3,4,5,6])
mental_states = ["Anxiety", "Bipolar", "Depression", "Normal", "Personality disorder", "Stress", "Suicidal"]

for c in le_classes:
    print(f"{c} : {mental_states[c]}")

# TF-IDF vektörleştirme (1 ve 2-gram)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train['tokens_stemmed'])
X_test_tfidf = vectorizer.transform(X_test['tokens_stemmed'])

# Sayısal özellikleri birleştirme
X_train_num = X_train[['num_of_characters', 'num_of_sentences']].values
X_test_num = X_test[['num_of_characters', 'num_of_sentences']].values
X_train_combined = hstack([X_train_tfidf, X_train_num])
X_test_combined = hstack([X_test_tfidf, X_test_num])

print('Number of feature words: ', len(vectorizer.get_feature_names_out()))
X_train_combined.shape

# Dengesiz sınıfları RandomOverSampler ile dengeleme
ros = RandomOverSampler(random_state=101)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_combined, y_train)
X_train_resampled.shape

# Kategorik hedef değişkenleri sayısal etiketlere dönüştür 
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)    
y_test_encoded = le.transform(y_test)          

# Model tanımları (XGBoost, LGBM, Naive Bayes, Logistic Regression vb.)
classifiers= {'XGB':XGBClassifier(learning_rate=0.2,max_depth=7,n_estimators=500,random_state=101,tree_method='gpu_hist'),
'Logistic Regression': LogisticRegression(solver='liblinear', penalty='l1', C=10, random_state=101),
'Naive Bayes':BernoulliNB(alpha=0.1, binarize=0.0),
'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=101),
'LightGBM':LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    learning_rate=0.1,
    n_estimators=500,
    max_depth=10,
    num_leaves=31),
}