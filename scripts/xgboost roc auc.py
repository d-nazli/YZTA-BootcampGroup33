# -*- coding: utf-8 -*-
"""
Created on Sun Aug  3 20:10:27 2025

@author: SOGUTPC
"""
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize

# 1. Etiketleri binarize et (her sınıf için one-vs-rest yaklaşımı)
y_test_binarized = label_binarize(y_test_encoded, classes=range(len(mental_states)))

# 2. XGBoost modeli için olasılık tahminlerini al
y_score = xgb_reg.predict_proba(X_test_combined)

# 3. ROC eğrisi ve AUC hesapla
plt.figure(figsize=(10, 8))
for i in range(len(mental_states)):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{mental_states[i]} (AUC = {roc_auc:.2f})')

# 4. Diyagonal çizgi ve grafiğin düzenlenmesi
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curve - XGBoost')
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
