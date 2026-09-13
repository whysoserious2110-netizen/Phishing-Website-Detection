import joblib
import pandas as pd

from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression

from feature_extractor_no_https import extract_url_features_no_https


# --------------------------------------------------
# Files
# --------------------------------------------------

MODEL_FILE = "hybrid_url_svm_v5.pkl"
VECTORIZER_FILE = "hybrid_url_svm_vectorizer_v5.pkl"
SCALER_FILE = "hybrid_url_svm_scaler_v5.pkl"

VALIDATION_FILE = "v5_validation.csv"

CALIBRATOR_FILE = "hybrid_url_svm_v5_calibrator.pkl"


# --------------------------------------------------
# Load V5 components
# --------------------------------------------------

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VECTORIZER_FILE)
scaler = joblib.load(SCALER_FILE)


# --------------------------------------------------
# Load validation set
# --------------------------------------------------

df = pd.read_csv(VALIDATION_FILE)

df = df.dropna(subset=["URL", "label"]).copy()

df["URL"] = df["URL"].astype(str)

y = df["label"].astype(int).to_numpy()


# --------------------------------------------------
# Character TF-IDF
# --------------------------------------------------

text_features = vectorizer.transform(
    df["URL"].tolist()
)


# --------------------------------------------------
# Numeric URL features
# --------------------------------------------------

numeric_features = [
    extract_url_features_no_https(url)
    for url in df["URL"]
]


numeric_df = pd.DataFrame(
    numeric_features
)


scaled_numeric = scaler.transform(
    numeric_df
)


# --------------------------------------------------
# Combine features
#
# IMPORTANT:
# Same order used by V5:
# TEXT + NUMERIC
# --------------------------------------------------

combined_features = hstack(
    [
        text_features,
        scaled_numeric
    ]
).tocsr()


# --------------------------------------------------
# Get V5 decision scores
# --------------------------------------------------

decision_scores = model.decision_function(
    combined_features
)


# --------------------------------------------------
# Calibrate decision scores
#
# V5 classes:
# 0 = phishing
# 1 = legitimate
#
# Positive SVM scores correspond to class 1.
# Therefore the calibrator learns:
#
# decision score -> P(legitimate)
# --------------------------------------------------

calibrator = LogisticRegression(
    max_iter=1000,
    random_state=42
)


calibrator.fit(
    decision_scores.reshape(-1, 1),
    y
)


# --------------------------------------------------
# Validation calibration probabilities
# --------------------------------------------------

legitimate_probability = calibrator.predict_proba(
    decision_scores.reshape(-1, 1)
)[:, 1]

phishing_probability = 1.0 - legitimate_probability


# --------------------------------------------------
# Save calibrator
# --------------------------------------------------

joblib.dump(
    calibrator,
    CALIBRATOR_FILE
)


# --------------------------------------------------
# Report
# --------------------------------------------------

print("=== V5 SVM PROBABILITY CALIBRATION ===")
print()
print("Validation rows:", len(df))
print("Phishing rows:", int((y == 0).sum()))
print("Legitimate rows:", int((y == 1).sum()))
print()
print("Calibration method: Logistic sigmoid")
print("Calibration source: V5 validation set")
print()
print("Saved:")
print(CALIBRATOR_FILE)
print()
print(
    "Example probability range:",
    round(float(phishing_probability.min()), 4),
    "to",
    round(float(phishing_probability.max()), 4)
)