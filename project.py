import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# --- 1. Load & clean ---
df = pd.read_csv("shot_logs.csv").dropna(subset=["SHOT_RESULT", "SHOT_CLOCK", "SHOT_DIST", "CLOSE_DEF_DIST"])

# --- 2. Feature engineering ---
def zone(dist):
    return pd.cut(dist, bins=[-1, 4, 10, 22, np.inf],labels=["Paint","Mid-Range","Long 2","3PT"])

def defense(dist):
    return pd.cut(dist, bins=[-1,2,4,np.inf],labels=["Tight","Moderate","Open"])

def clock_phase(clock):
    return pd.cut(clock, bins=[0,7,14,np.inf],labels=["Late","Mid","Early"])

df["SHOT_ZONE"]= zone(df["SHOT_DIST"])
df["DEFENSE"]= defense(df["CLOSE_DEF_DIST"])
df["SHOT_CLOCK_PHASE"]= clock_phase(df["SHOT_CLOCK"])
df["TARGET"]= (df["SHOT_RESULT"]=="made").astype(int)



# --- 3. Train/test split ---
X = df[["SHOT_DIST","SHOT_CLOCK","CLOSE_DEF_DIST","SHOT_ZONE","DEFENSE","SHOT_CLOCK_PHASE"]]
y = df["TARGET"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Build preprocessing + model pipeline ---
numeric_feats   = ["SHOT_DIST","SHOT_CLOCK","CLOSE_DEF_DIST"]
categorical_feats = ["SHOT_ZONE","DEFENSE","SHOT_CLOCK_PHASE"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric_feats),
    ("cat", OneHotEncoder(drop="first"), categorical_feats)
])

xgb_clf = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

pipeline = ImbPipeline(steps=[
    ("prep", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", xgb_clf)
])

# --- 5. Hyperparameter tuning ---
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1],
    "model__colsample_bytree": [0.8, 1],
}
grid = GridSearchCV(
    pipeline, param_grid,
    cv=5, scoring="f1", n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# --- 6. Evaluation fn ---
def evaluate(m, X, y, label):
    y_pred  = m.predict(X)
    y_proba = m.predict_proba(X)[:,1]
    print(f"{label}")
    print(f"  Accuracy:  {accuracy_score(y,y_pred):.3f}")
    print(f"  Precision: {precision_score(y,y_pred):.3f}")
    print(f"  Recall:    {recall_score(y,y_pred):.3f}")
    print(f"  F1:        {f1_score(y,y_pred):.3f}")
    print(f"  ROC-AUC:   {roc_auc_score(y,y_proba):.3f}\n")

print("\nTest Set Performance:")
evaluate(best_model, X_test, y_test, "XGBoost")

# --- 7. Plots (simplified) ---
# Feature importance
importances = best_model.named_steps["model"].feature_importances_
feat_names = (numeric_feats +
              list(best_model.named_steps["prep"]
                   .named_transformers_["cat"]
                   .get_feature_names_out(categorical_feats)))
plt.figure(figsize=(8,5))
plt.barh(feat_names, importances)
plt.title("Feature Importance")
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

# Confusion matrix
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap="Blues", interpolation="nearest")
plt.title("Confusion Matrix")
plt.xticks([0,1],["Miss","Make"])
plt.yticks([0,1],["Miss","Make"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i,j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(
    y_test, best_model.predict_proba(X_test)[:,1]
)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0,1],[0,1],"--", label="Chance")
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.tight_layout()
plt.show()
