cat > README.md <<'EOF'
# PhishGuard – Phishing Website Detection System

PhishGuard is a machine-learning-based phishing website detection system that analyzes a URL and classifies it as either a potential phishing website or a likely legitimate website.

## Features

- URL-based phishing detection
- Character-level TF-IDF analysis
- 17 numerical URL features
- Logistic Regression classification
- Phishing and legitimate probability scores
- Human-readable prediction explanations
- Flask-based web interface

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Joblib
- HTML/CSS
- UCI PhiUSIIL Phishing URL Dataset
- PhishTank
- Tranco

## Machine Learning Approach

The final PhishGuard model uses a hybrid URL representation consisting of:

1. Character-level TF-IDF features extracted from the URL.
2. 17 numerical URL features describing characteristics such as URL length, domain length, subdomain count, digits, special characters, and obfuscation.
3. Logistic Regression for classification.

The final application uses:

- `hybrid_url_model_v4.pkl`
- `hybrid_url_vectorizer_v4.pkl`
- `hybrid_url_scaler_v4.pkl`

## Final Model

V4 was selected after comparing multiple experimental versions.

| Model | Accuracy | Precision | Recall | F1-Score |
|------|---------:|----------:|-------:|---------:|
| V3 | 99.37% | 99.68% | 98.86% | 99.27% |
| V4 | 99.39% | 99.74% | 98.87% | 99.30% |
| V5 | 91.96% | 95.13% | 86.01% | 90.34% |

V4 provided the strongest overall internal performance.

## External Evaluation

A corrected primary external evaluation used 114 URLs:

- 57 legitimate URLs
- 57 phishing URLs
- 0 URL overlap with training data
- 0 domain overlap with training data

V4 achieved on this evaluation set:

- Accuracy: 100%
- Precision: 100%
- Recall: 100%
- F1-Score: 100%
- Phishing detection: 57/57
- Legitimate acceptance: 57/57

An additional unseen PhishTank phishing stress test detected:

- 68/68 phishing URLs

These results apply only to the constructed evaluation datasets and should not be interpreted as universal real-world accuracy.

## How to Run

Create a Python virtual environment:

```bash
python -m venv venv
