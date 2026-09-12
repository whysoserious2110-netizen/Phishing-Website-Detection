import pandas as pd
import joblib

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from feature_extractor_no_https import extract_url_features_no_https


# ===================================
# LOAD V4 DATASET
# ===================================

df = pd.read_csv("mixed_training_data_v4.csv")

print("===================================")
print("HYBRID MODEL V4 TRAINING")
print("===================================")
print()

print("Rows:", len(df))
print("Class distribution:")
print(df["label"].value_counts())
print()


# ===================================
# EXTRACT NUMERIC FEATURES
# ===================================

print("Extracting numeric URL features...")

numeric_features = df["URL"].apply(
    extract_url_features_no_https
).tolist()

numeric_df = pd.DataFrame(numeric_features)

print("Numeric feature shape:", numeric_df.shape)
print()


# ===================================
# CHARACTER TF-IDF
# SAME SETTINGS AS V3
# ===================================

print("Building character TF-IDF features...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=150000,
    sublinear_tf=True
)

text_features = vectorizer.fit_transform(df["URL"])

print("Character feature shape:", text_features.shape)
print()


# ===================================
# SCALE NUMERIC FEATURES
# ===================================

scaler = StandardScaler()

numeric_scaled = scaler.fit_transform(numeric_df)

print("Numeric features scaled.")
print()


# ===================================
# COMBINE FEATURES
# IMPORTANT:
# V3 ORDER = TEXT + NUMERIC
# ===================================

X = hstack([
    text_features,
    numeric_scaled
])

y = df["label"]

print("Combined feature shape:", X.shape)
print()


# ===================================
# TRAIN / TEST SPLIT
# SAME AS V3
# ===================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print()


# ===================================
# TRAIN MODEL
# SAME AS V3
# ===================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=2.0,
    solver="liblinear",
    random_state=42
)

print("Training V4 model...")
model.fit(X_train, y_train)

print("Training complete.")
print()


# ===================================
# EVALUATE
# ===================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    pos_label=0
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label=0
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label=0
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)


print("===================================")
print("HYBRID MODEL V4 RESULTS")
print("===================================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")

print()
print("Confusion Matrix:")
print(cm)

print()
print("Class meaning:")
print("0 = Phishing")
print("1 = Legitimate")

print()


# ===================================
# SAVE V4 MODEL
# ===================================

joblib.dump(
    model,
    "hybrid_url_model_v4.pkl"
)

joblib.dump(
    vectorizer,
    "hybrid_url_vectorizer_v4.pkl"
)

joblib.dump(
    scaler,
    "hybrid_url_scaler_v4.pkl"
)


print("Saved:")
print("hybrid_url_model_v4.pkl")
print("hybrid_url_vectorizer_v4.pkl")
print("hybrid_url_scaler_v4.pkl")