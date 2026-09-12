from flask import Flask, render_template, request
import joblib
import pandas as pd

from scipy.sparse import hstack, csr_matrix

from feature_extractor_no_https import (
    extract_url_features_no_https
)

from url_normalizer import normalize_url


app = Flask(__name__)


# --------------------------------------------------
# Load V4 hybrid model components
# --------------------------------------------------

model = joblib.load(
    "hybrid_url_model_v4.pkl"
)

vectorizer = joblib.load(
    "hybrid_url_vectorizer_v4.pkl"
)

scaler = joblib.load(
    "hybrid_url_scaler_v4.pkl"
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
        numeric_df
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


    probabilities = model.predict_proba(
        combined_features
    )[0]


    # Model class:
    # 0 = phishing
    # 1 = legitimate

    phishing_probability = probabilities[
        list(model.classes_).index(0)
    ]


    legitimate_probability = probabilities[
        list(model.classes_).index(1)
    ]


    # ----------------------------------------------
    # Result
    # ----------------------------------------------

    if prediction == 0:

        result = "Potential Phishing Website"

    else:

        result = "Likely Legitimate Website"


    # ----------------------------------------------
    # Explanation
    # ----------------------------------------------

    explanation = generate_explanation(
        prediction,
        phishing_probability,
        numeric_features,
        url
    )


    return (
        result,
        phishing_probability,
        legitimate_probability,
        numeric_features,
        explanation
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
                explanation
            ) = predict_url(url)


    return render_template(
        "index.html",
        result=result,
        url=url,
        phishing_probability=phishing_probability,
        legitimate_probability=legitimate_probability,
        features=features,
        explanation=explanation
    )


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )