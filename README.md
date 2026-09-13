# PhishGuard - Phishing Website Detection System

PhishGuard is a machine-learning-based phishing website detection system that analyzes a URL and classifies it as either a potential phishing website or a likely legitimate website.

## Features

- URL-based phishing detection
- Character-level TF-IDF analysis
- 17 numerical URL features
- Linear Support Vector Machine (Linear SVM)
- Calibrated phishing and legitimate probability estimates
- Human-readable prediction explanations
- Risk assessment
- Flask-based web interface

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Joblib
- BeautifulSoup4 for HTML research experiments
- UCI PhiUSIIL Phishing URL Dataset
- PhishTank
- Tranco

## Machine Learning Approach

The final PhishGuard model uses a hybrid URL representation consisting of:

1. Character-level TF-IDF features extracted from the URL.
2. 17 numerical URL features describing URL length, domain length, subdomain count, digits, special characters, and obfuscation.
3. A Linear Support Vector Machine (LinearSVC) for classification.

The final application uses a separate calibration model trained on the validation set to convert the SVM decision score into a probability estimate.

The final application uses:

- `hybrid_url_svm_v5.pkl`
- `hybrid_url_svm_vectorizer_v5.pkl`
- `hybrid_url_svm_scaler_v5.pkl`
- `hybrid_url_svm_v5_calibrator.pkl`

## Final Model

Several model versions and algorithms were evaluated during development.

The final V5 Linear SVM was selected because it provided the strongest performance under a registrable-domain-disjoint train/validation/test split.

### V5 Unseen-Domain Test Results

| Metric | Result |
|---|---:|
| Accuracy | 99.46% |
| Precision | 99.86% |
| Phishing Recall | 98.82% |
| F1-Score | 99.34% |
| Legitimate Acceptance | 99.90% |

The V5 test set was created using a registrable-domain group split. No registrable domains, hostnames, or exact URLs were shared between the training, validation, and test partitions.

## Model Comparison

Under the same V5 split, the evaluated models were:

| Model | Test Accuracy | Precision | Phishing Recall | F1-Score |
|---|---:|---:|---:|---:|
| Linear SVM | 99.46% | 99.86% | 98.82% | 99.34% |
| Logistic Regression | 99.23% | 99.71% | 98.41% | 99.05% |
| Random Forest | 97.77% | 98.07% | 96.49% | 97.27% |

The Linear SVM was therefore selected as the final URL-based classifier.

## Error Analysis

The final V5 test set contained 192 missed phishing URLs and 23 legitimate URLs incorrectly classified as phishing.

Further analysis showed that:

- 58.33% of the missed phishing URLs belonged to a simple-URL subset defined using multiple low-complexity URL characteristics.
- The missed phishing URLs were generally shorter and contained fewer digits, subdomains, query parameters, and special characters than phishing URLs detected correctly.
- Missed phishing URLs and legitimate false positives had substantial overlap in their URL structure.
- This indicates that some phishing URLs are difficult to distinguish from legitimate URLs using URL syntax alone.

This limitation was retained rather than addressed through hardcoded rules.

## HTML Retrieval Study

A separate HTML feasibility experiment was conducted using a small balanced sample.

The experiment found substantial retrieval bias:

| | Phishing | Legitimate |
|---|---:|---:|
| HTML available | 23.67% | 85.67% |
| HTTP 200 responses | 12.67% | 68.00% |
| Connection errors | 70.33% | 6.00% |

Because live HTML retrieval was strongly biased by availability and server behavior, and because some downloaded webpages triggered local antivirus detections, the raw HTML corpus was not used to train the final model.

## Threat-Intelligence Evaluation

External phishing intelligence was investigated as a possible enhancement.

A retrospective comparison with the existing PhishTank dataset showed that PhishTank could potentially cover some difficult cases, but the existing PhishTank data had already contributed to the training-data construction. Therefore, those results were not treated as an independent model improvement.

A separate OpenPhish Community Feed experiment was also performed.

For the V5 hard-case benchmark:

- 163 unique phishing domains were evaluated.
- 0 exact URL matches were found.
- 0 hostname matches were found.
- 2 registrable-domain matches were found, both involving shared hosting-platform domains.

The OpenPhish snapshot therefore provided insufficient coverage to justify adding it to the final model.

## Safety and Research Limitations

PhishGuard is a research prototype and should not be treated as a complete phishing-detection or cybersecurity solution.

Important limitations include:

- URL-only detection cannot reliably determine the true intent of every website.
- Some legitimate URLs may be classified as phishing.
- Some phishing URLs may be classified as legitimate.
- The reported test metrics apply to the constructed domain-disjoint evaluation dataset.
- They should not be interpreted as universal real-world accuracy.
- Reputation and live HTML signals were investigated but were not incorporated into the final model because of coverage, leakage, retrieval-bias, and safety concerns.

## How to Run

Create a Python virtual environment:

```bash
python -m venv venv