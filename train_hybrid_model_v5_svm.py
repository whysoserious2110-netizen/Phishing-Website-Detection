import pandas as pd
import joblib

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from feature_extractor_no_https import extract_url_features_no_https


TRAIN_FILE = "v5_train.csv"
VALIDATION_FILE = "v5_validation.csv"
TEST_FILE = "v5_test.csv"


print("===================================")
print("V5 HYBRID LINEAR SVM BENCHMARK")
print("DOMAIN-DISJOINT EVALUATION")
print("===================================")
print()


# ===================================
# 1. LOAD SPLITS
# ===================================

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)
test_df = pd.read_csv(TEST_FILE)

print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Test rows:", len(test_df))
print()


# ===================================
# 2. EXTRACT NUMERIC FEATURES
# ===================================

print("Extracting numeric URL features...")

train_numeric = train_df["URL"].apply(
    extract_url_features_no_https
).tolist()

validation_numeric = validation_df["URL"].apply(
    extract_url_features_no_https
).tolist()

test_numeric = test_df["URL"].apply(
    extract_url_features_no_https
).tolist()


X_train_numeric = pd.DataFrame(train_numeric)
X_validation_numeric = pd.DataFrame(validation_numeric)
X_test_numeric = pd.DataFrame(test_numeric)


y_train = train_df["label"]
y_validation = validation_df["label"]
y_test = test_df["label"]


print(
    "Training numeric shape:",
    X_train_numeric.shape
)

print(
    "Validation numeric shape:",
    X_validation_numeric.shape
)

print(
    "Test numeric shape:",
    X_test_numeric.shape
)

print()


# ===================================
# 3. FIT TF-IDF ON TRAINING ONLY
# ===================================

print("Training TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=150000,
    sublinear_tf=True
)


X_train_text = vectorizer.fit_transform(
    train_df["URL"]
)

X_validation_text = vectorizer.transform(
    validation_df["URL"]
)

X_test_text = vectorizer.transform(
    test_df["URL"]
)


print(
    "Training text shape:",
    X_train_text.shape
)

print(
    "Validation text shape:",
    X_validation_text.shape
)

print(
    "Test text shape:",
    X_test_text.shape
)

print()


# ===================================
# 4. SCALE NUMERIC FEATURES
# ===================================

print("Scaling numeric features...")

scaler = StandardScaler()

X_train_numeric_scaled = scaler.fit_transform(
    X_train_numeric
)

X_validation_numeric_scaled = scaler.transform(
    X_validation_numeric
)

X_test_numeric_scaled = scaler.transform(
    X_test_numeric
)


# ===================================
# 5. COMBINE FEATURES
# ===================================

print("Combining TEXT + NUMERIC features...")

X_train = hstack([
    X_train_text,
    X_train_numeric_scaled
]).tocsr()

X_validation = hstack([
    X_validation_text,
    X_validation_numeric_scaled
]).tocsr()

X_test = hstack([
    X_test_text,
    X_test_numeric_scaled
]).tocsr()


print(
    "Combined training shape:",
    X_train.shape
)

print(
    "Combined validation shape:",
    X_validation.shape
)

print(
    "Combined test shape:",
    X_test.shape
)

print()


# ===================================
# 6. TRAIN LINEAR SVM
# ===================================

print("Training Linear SVM...")

model = LinearSVC(
    C=2.0,
    class_weight="balanced",
    max_iter=5000,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("Training complete.")
print()


# ===================================
# 7. VALIDATION
# ===================================

validation_pred = model.predict(
    X_validation
)

validation_accuracy = accuracy_score(
    y_validation,
    validation_pred
)

validation_precision = precision_score(
    y_validation,
    validation_pred,
    pos_label=0
)

validation_recall = recall_score(
    y_validation,
    validation_pred,
    pos_label=0
)

validation_f1 = f1_score(
    y_validation,
    validation_pred,
    pos_label=0
)

validation_cm = confusion_matrix(
    y_validation,
    validation_pred,
    labels=[0, 1]
)


# ===================================
# 8. TEST
# ===================================

test_pred = model.predict(
    X_test
)

test_accuracy = accuracy_score(
    y_test,
    test_pred
)

test_precision = precision_score(
    y_test,
    test_pred,
    pos_label=0
)

test_recall = recall_score(
    y_test,
    test_pred,
    pos_label=0
)

test_f1 = f1_score(
    y_test,
    test_pred,
    pos_label=0
)

test_cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1]
)


# ===================================
# 9. VALIDATION RESULTS
# ===================================

print("===================================")
print("LINEAR SVM VALIDATION RESULTS")
print("===================================")

print(
    f"Accuracy:  {validation_accuracy:.4f}"
)

print(
    f"Precision: {validation_precision:.4f}"
)

print(
    f"Recall:    {validation_recall:.4f}"
)

print(
    f"F1:        {validation_f1:.4f}"
)

print()
print("Confusion Matrix:")
print(validation_cm)
print()


# ===================================
# 10. TEST RESULTS
# ===================================

print("===================================")
print("LINEAR SVM UNSEEN-DOMAIN TEST")
print("===================================")

print(
    f"Accuracy:  {test_accuracy:.4f}"
)

print(
    f"Precision: {test_precision:.4f}"
)

print(
    f"Recall:    {test_recall:.4f}"
)

print(
    f"F1:        {test_f1:.4f}"
)

print()
print("Confusion Matrix:")
print(test_cm)
print()


# ===================================
# 11. PHISHING DETECTION RATE
# ===================================

phishing_total = (
    y_test == 0
).sum()

phishing_detected = (
    (y_test == 0) &
    (test_pred == 0)
).sum()

phishing_detection_rate = (
    phishing_detected / phishing_total
    if phishing_total > 0
    else 0
)


# ===================================
# 12. LEGITIMATE ACCEPTANCE RATE
# ===================================

legitimate_total = (
    y_test == 1
).sum()

legitimate_accepted = (
    (y_test == 1) &
    (test_pred == 1)
).sum()

legitimate_acceptance_rate = (
    legitimate_accepted / legitimate_total
    if legitimate_total > 0
    else 0
)


print("===================================")
print("LINEAR SVM TEST RATES")
print("===================================")

print(
    f"Phishing detection rate: "
    f"{phishing_detection_rate:.4f}"
)

print(
    f"Legitimate acceptance rate: "
    f"{legitimate_acceptance_rate:.4f}"
)

print()


# ===================================
# 13. SAVE MODEL
# ===================================

joblib.dump(
    model,
    "hybrid_url_svm_v5.pkl"
)

joblib.dump(
    vectorizer,
    "hybrid_url_svm_vectorizer_v5.pkl"
)

joblib.dump(
    scaler,
    "hybrid_url_svm_scaler_v5.pkl"
)


print("===================================")
print("LINEAR SVM FILES SAVED")
print("===================================")

print("hybrid_url_svm_v5.pkl")
print("hybrid_url_svm_vectorizer_v5.pkl")
print("hybrid_url_svm_scaler_v5.pkl")

print()
print("===================================")
print("LINEAR SVM BENCHMARK COMPLETE")
print("===================================")