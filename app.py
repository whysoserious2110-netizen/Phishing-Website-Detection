from flask import Flask, render_template, request
import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from feature_extractor_no_https import (
    extract_url_features_no_https
)

from url_normalizer import normalize_url
from domain_analyzer import analyze_domain
from risk_engine import calculate_risk


app = Flask(__name__)


# --------------------------------------------------
# Load V5 hybrid SVM model components
# --------------------------------------------------

model = joblib.load(
    "hybrid_url_svm_v5.pkl"
)

vectorizer = joblib.load(
    "hybrid_url_svm_vectorizer_v5.pkl"
)

scaler = joblib.load(
    "hybrid_url_svm_scaler_v5.pkl"
)

calibrator = joblib.load(
    "hybrid_url_svm_v5_calibrator.pkl"
)


# --------------------------------------------------
# Feature names
# --------------------------------------------------

feature_names = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL"
]


# --------------------------------------------------
# Generate explanation
# --------------------------------------------------

def generate_explanation(
    prediction,
    phishing_probability,
    features,
    url
):

    (
        url_length,
        domain_length,
        is_domain_ip,
        tld_length,
        subdomains,
        has_obfuscation,
        obfuscated_chars,
        obfuscation_ratio,
        letters,
        letter_ratio,
        digits,
        digit_ratio,
        equals,
        qmarks,
        ampersands,
        special_chars,
        special_char_ratio
    ) = features


    # ----------------------------------------------
    # Phishing explanation
    # ----------------------------------------------

    if prediction == 0:

        reasons = []

        if is_domain_ip == 1:
            reasons.append(
                "the domain is represented by an IP address"
            )

        if url_length >= 75:
            reasons.append(
                "the URL is unusually long"
            )

        if domain_length >= 30:
            reasons.append(
                "the domain name is unusually long"
            )

        if subdomains >= 2:
            reasons.append(
                "the URL contains multiple subdomains"
            )

        if digits >= 5:
            reasons.append(
                "the URL contains a relatively high number of digits"
            )

        if special_chars >= 8:
            reasons.append(
                "the URL contains many special characters"
            )

        if qmarks >= 1:
            reasons.append(
                "the URL contains query parameters"
            )

        if equals >= 1:
            reasons.append(
                "the URL contains parameter assignments"
            )

        if has_obfuscation == 1:
            reasons.append(
                "the URL contains encoded or obfuscated characters"
            )

        if not reasons:
            reasons.append(
                "the URL pattern matches patterns learned from phishing examples"
            )

        reason_text = ", ".join(
            reasons[:4]
        )

        explanation = (
            "The model identified this URL as potentially phishing. "
            "The prediction is influenced by "
            + reason_text
            + ". "
            "These characteristics can occur on legitimate websites as well, "
            "so the prediction should be treated as a warning rather than proof "
            "that the website is malicious. Avoid entering sensitive information "
            "until the website can be independently verified."
        )


    # ----------------------------------------------
    # Legitimate explanation
    # ----------------------------------------------

    else:

        reasons = []

        if url_length <= 40:
            reasons.append(
                "the URL is relatively short"
            )

        if subdomains <= 1:
            reasons.append(
                "the URL has a simple subdomain structure"
            )

        if digits == 0:
            reasons.append(
                "the URL contains no digits"
            )

        if has_obfuscation == 0:
            reasons.append(
                "no URL obfuscation was detected"
            )

        if special_chars <= 5:
            reasons.append(
                "the URL contains relatively few special characters"
            )

        if not reasons:
            reasons.append(
                "the URL pattern is closer to legitimate examples learned by the model"
            )

        reason_text = ", ".join(
            reasons[:4]
        )

        explanation = (
            "The model classified this URL as likely legitimate. "
            "The prediction is influenced by "
            + reason_text
            + ". "
            "However, this does not guarantee that the website is completely safe. "
            "Always verify the domain and website before entering passwords, "
            "payment information, or other sensitive data."
        )


    return explanation


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_url(url):

    # ----------------------------------------------
    # Normalize URL
    # ----------------------------------------------

    url = normalize_url(url)


    # ----------------------------------------------
    # Character-level TF-IDF
    # ----------------------------------------------

    text_features = vectorizer.transform(
        [url]
    )


    # ----------------------------------------------
    # Numeric URL features
    # ----------------------------------------------

    numeric_features = extract_url_features_no_https(
        url
    )


    numeric_df = pd.DataFrame(
        [numeric_features],
        columns=feature_names
    )


    scaled_numeric = scaler.transform(
    numeric_df.to_numpy()
)


    scaled_numeric = csr_matrix(
        scaled_numeric
    )


    # ----------------------------------------------
    # Combine features
    #
    # V4 training order:
    # TEXT + NUMERIC
    # ----------------------------------------------

    combined_features = hstack(
        [
            text_features,
            scaled_numeric
        ]
    ).tocsr()


    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    prediction = int(
        model.predict(
            combined_features
        )[0]
    )


    # --------------------------------------------------
# Convert SVM decision score to calibrated probability
# --------------------------------------------------

    decision_score = model.decision_function(
        combined_features
    )[0]

    legitimate_probability = calibrator.predict_proba(
        [[decision_score]]
    )[0, 1]

    phishing_probability = 1.0 - legitimate_probability


    # Model class:
    # 0 = phishing
    # 1 = legitimate


    # ----------------------------------------------
    # Result
    # ----------------------------------------------

    if prediction == 0:

        result = "Potential Phishing Website"

    else:

        result = "Likely Legitimate Website"


    # ----------------------------------------------
    # Domain analysis
    # ----------------------------------------------

    domain_info = analyze_domain(
        url
    )


    is_ip_address = domain_info[
        "is_ip_address"
    ]

    subdomain_count = domain_info[
        "subdomain_count"
    ]


    # ----------------------------------------------
    # Risk assessment
    # ----------------------------------------------

    risk_assessment = calculate_risk(
        prediction,
        phishing_probability,
        legitimate_probability,
        is_ip_address,
        subdomain_count
    )


    risk_level = risk_assessment[
        "risk_level"
    ]

    risk_reason = risk_assessment[
        "reason"
    ]


    # ----------------------------------------------
    # Explanation
    # ----------------------------------------------

    explanation = generate_explanation(
        prediction,
        phishing_probability,
        numeric_features,
        url
    )


    # ----------------------------------------------
    # Return results
    # ----------------------------------------------

    return (
        result,
        phishing_probability,
        legitimate_probability,
        numeric_features,
        explanation,
        risk_level,
        risk_reason
    )


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    result = None
    url = ""

    phishing_probability = None
    legitimate_probability = None

    features = None
    explanation = None

    risk_level = None
    risk_reason = None


    if request.method == "POST":

        url = request.form.get(
            "url",
            ""
        ).strip()


        if url:

            (
                result,
                phishing_probability,
                legitimate_probability,
                features,
                explanation,
                risk_level,
                risk_reason
            ) = predict_url(url)


    return render_template(
        "index.html",
        result=result,
        url=url,
        phishing_probability=phishing_probability,
        legitimate_probability=legitimate_probability,
        features=features,
        explanation=explanation,
        risk_level=risk_level,
        risk_reason=risk_reason
    )


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )