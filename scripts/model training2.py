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
    num_leaves=31)
}

#Model eğitimleri,tahminler,doğruluk skorları ve raporları
print("\nTraining XGBoost...")
xgb_reg = classifiers['XGB']
xgb_reg.fit(X_train_resampled, y_train_resampled) 
y_pred = xgb_reg.predict(X_test_combined)

print("\nXGBoost Results:")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
print(classification_report(y_test_encoded, y_pred,target_names=mental_states))


print("\nTraining Logistic Regression...")
log_reg = classifiers['Logistic Regression']
log_reg.fit(X_train_resampled, y_train_resampled) 
y_pred_log = log_reg.predict(X_test_combined) 

print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_log))
print(classification_report(y_test_encoded, y_pred_log,target_names=mental_states))


print("\nTraining Naive Bayes...")
naive_bayes = classifiers['Naive Bayes']
naive_bayes.fit(X_train_resampled, y_train_resampled) 
y_pred_naive = naive_bayes.predict(X_test_combined) 

print("\nNaive Bayes Results:")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_naive))
print(classification_report(y_test_encoded, y_pred_naive, target_names=mental_states))

print("\nTraining Random Forest...")
random_forest = classifiers['Random Forest']
random_forest.fit(X_train_resampled,y_train_resampled)
y_pred_random = random_forest.predict(X_test_combined)

print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_random))
print(classification_report(y_test_encoded, y_pred_random, target_names=mental_states))

print("\nTraining LightGBM...")
lgb_reg = classifiers['LightGBM']
lgb_reg.fit(X_train_resampled, y_train_resampled) # X_train_tfidf yerine X_train_resampled
y_pred_lgb = lgb_reg.predict(X_test_combined) # X_test_tfidf yerine X_test_combined

print("\nLightGBM Results:")
print("Accuracy:", accuracy_score(y_test_encoded, y_pred_lgb))
print(classification_report(y_test_encoded, y_pred_lgb,target_names=mental_states))

accuracy_scores = [
    ('XGBoost', accuracy_score(y_test_encoded, y_pred)),
    ('Logistic Regression', accuracy_score(y_test_encoded, y_pred_log)),
    ('Naive Bayes', accuracy_score(y_test_encoded, y_pred_naive)),
    ('Random Forest', accuracy_score(y_test_encoded, y_pred_random)),
    ('LightGBM', accuracy_score(y_test_encoded, y_pred_lgb))
]

predictions = {
    'XGBoost': y_pred,
    'Logistic Regression': y_pred_log,
    'Naive Bayes': y_pred_naive,
    'Random Forest':y_pred_random,
    'LightGBM':y_pred_lgb
}

# Confussion matrix görselleri
for name, y_pred in predictions.items():
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=mental_states, yticklabels=mental_states)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {name}')
    plt.tight_layout()
    plt.show()
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    accuracy_scores.append((name, accuracy)) 

# Accuracy skorlarını barplot ile çizdirme
accuracies_df = pd.DataFrame(accuracy_scores, columns=["Classifier", "Accuracy"]).sort_values("Accuracy", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Accuracy", y="Classifier", data=accuracies_df, palette="viridis")
plt.title("Classifier Accuracy Comparison")
plt.xlim(0, 1)
plt.xlabel("Accuracy Score")
plt.ylabel("Classifier")
plt.tight_layout()
plt.show()

#Model performans metriklerinin karşılaştırması
metrics = []
for name, y_pred in predictions.items():
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)

    metrics.append({
        'Classifier': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })
  
metrics_df = pd.DataFrame(metrics)
metrics_long = metrics_df.melt(id_vars='Classifier',
                                value_vars=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                var_name='Metric',
                                value_name='Score')

plt.figure(figsize=(12, 6))
sns.barplot(x='Score', y='Classifier', hue='Metric', data=metrics_long, palette='Set1')

plt.title("Model Performance Comparison (Accuracy, Precision, Recall, F1 Score)")
plt.xlim(0, 1.05)
plt.xlabel("Score")
plt.ylabel("Classifier")
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

