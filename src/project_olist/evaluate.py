from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from project_olist.data import y_test
from sklearn.metrics import classification_report
from project_olist.data import X_test
from project_olist.config import MODEL_PATH
import joblib

model = joblib.load(MODEL_PATH)

y_pred = model.predict(X_test)

print(f"""
Classification Report:

{classification_report(y_test, y_pred)}
""")

print(f"""
ROC Auc:

{roc_auc_score(y_test, y_pred)}
""")

print(f"""
Confusion Matrix:

{confusion_matrix(y_test, y_pred)}
""")