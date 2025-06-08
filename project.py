# Shot prediction ML project - using real shot log data to classify if a shot is made or missed.
# Wanted to explore what factors (distance, time, defender, etc.) really affect shot outcomes.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Standard scikit-learn stuff for modeling + evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import ( accuracy_score, precision_score, recall_score,f1_score, roc_auc_score, confusion_matrix, roc_curve)

# I went with XGBoost because it's generally great for tabular data and handles imbalanced sets pretty well
from xgboost import XGBClassifier

# SMOTE to handle class imbalance (way more missed shots than made ones)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# === 1. Load + Clean ===

# First step — just loading the data and dropping rows with missing key features
# I only dropped rows where missing values directly affect the features I plan to use
df = pd.read_csv("shot_logs.csv")
df = df.dropna(subset=["SHOT_RESULT", "SHOT_CLOCK", "SHOT_DIST", "CLOSE_DEF_DIST"])

# === 2. Feature Engineering ===

# I created shot zones manually instead of relying on court zones, because I wanted more control over how ranges are grouped (e.g., Paint vs Long 2s)
def zone(dist):
    return pd.cut(dist, bins=[-1, 4, 10, 22, np.inf], labels=["Paint", "Mid-Range", "Long 2", "3PT"])

# Defender distance also turned into categories, tight defense tends to matter more,so wanted to capture that as a feature
def defense(dist):
    return pd.cut(dist, bins=[-1, 2, 4, np.inf], labels=["Tight", "Moderate", "Open"])

# Shot clock phase — I broke this into early/mid/late clock because
# shot quality often drops at the end of the clock and I wanted to test that pattern
def clock_phase(clock):
    return pd.cut(clock, bins=[0, 7, 14, np.inf], labels=["Late", "Mid", "Early"])

# Apply all the custom features
df["SHOT_ZONE"] = zone(df["SHOT_DIST"])
df["DEFENSE"] = defense(df["CLOSE_DEF_DIST"])
df["SHOT_CLOCK_PHASE"] = clock_phase(df["SHOT_CLOCK"])

# Create binary target — 1 for made, 0 for missed
df["TARGET"] = (df["SHOT_RESULT"] == "made").astype(int)

# === 3. Train/Test Split ===

# Selected features I think matter most based on intuition and availability in the dataset
X = df[["SHOT_DIST", "SHOT_CLOCK", "CLOSE_DEF_DIST", "SHOT_ZONE", "DEFENSE", "SHOT_CLOCK_PHASE"]]
y = df["TARGET"]

# Stratified split so the train/test sets have the same made/miss ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === 4. Preprocessing + Model Pipeline ===

# Broke features into numeric vs categorical manually so I have full control over preprocessing
num_cols = ["SHOT_DIST", "SHOT_CLOCK", "CLOSE_DEF_DIST"]
cat_cols = ["SHOT_ZONE", "DEFENSE", "SHOT_CLOCK_PHASE"]

# For numerics: scaling with StandardScaler (XGBoost doesn't strictly need it but it's good practice)
# For categoricals: one-hot encode but drop first to avoid multicollinearity
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(drop="first"), cat_cols)
])

# XGBoost works really well for this task — robust, handles trees + interactions, and doesn't need too much tuning
xgb = XGBClassifier(use_label_encoder=False,eval_metric="logloss", random_state=42)

# Final pipeline includes:
# - Preprocessing
# - SMOTE to balance classes (this improved my recall a lot)
# - Classifier
pipeline = ImbPipeline(steps=[
    ("prep", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", xgb)
])

# === 5. Hyperparameter Tuning ===

# These are common XGBoost parameters and I limited the search space to keep training time reasonable
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.05, 0.1],
    "model__subsample": [0.8, 1],
    "model__colsample_bytree": [0.8, 1],
}

# F1 score is used because I care about both precision and recall. Especially since 'made' shots are less common
grid = GridSearchCV(
    pipeline, param_grid,
    cv=5, scoring="f1", verbose=1, n_jobs=-1
)

# Train the model with the best hyperparameters
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# === 6. Evaluation ===

# Simple utility to output all relevant metricsand is Helpful to track performance consistently when testing changes
def evaluate(model, X, y, label="Model"):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    print(f"\n{label} performance:")
    print(f"Accuracy:{accuracy_score(y, y_pred):.3f}")
    print(f"Precision: {precision_score(y, y_pred):.3f}")
    print(f"Recall:{recall_score(y, y_pred):.3f}")
    print(f"F1 Score:{f1_score(y, y_pred):.3f}")
    print(f"ROC AUC:{roc_auc_score(y, y_proba):.3f}")

# Evaluate on the test set
evaluate(best_model, X_test, y_test, "XGBoost Final")

# === 7. Visualizations ===

# Plotting feature importances 
# I wanted to see which features XGBoost found most helpful
feat_names = num_cols + list(
    best_model.named_steps["prep"]
    .named_transformers_["cat"]
    .get_feature_names_out(cat_cols)
)
importances = best_model.named_steps["model"].feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(feat_names, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Plotting confusion matrix to visualize actual vs predicted shot results
cm = confusion_matrix(y_test, best_model.predict(X_test))
plt.figure(figsize=(4, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks([0, 1], ["Miss", "Make"])
plt.yticks([0, 1], ["Miss", "Make"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Final ROC curve to show how well the classifier distinguishes between made and missed shots
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label="XGBoost")
plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()
