def calculate_risk(
    ml_prediction,
    phishing_probability,
    legitimate_probability,
    is_ip_address=False,
    subdomain_count=0
):
    if phishing_probability >= 0.75:
        if is_ip_address:
            return {
                "risk_level": "HIGH RISK",
                "reason": (
                    "The machine-learning model detected a high probability "
                    "of phishing, and the URL uses an IP address instead of "
                    "a normal domain name."
                ),
            }

        if subdomain_count >= 3:
            return {
                "risk_level": "HIGH RISK",
                "reason": (
                    "The machine-learning model detected a high probability "
                    "of phishing, and the URL contains multiple subdomains."
                ),
            }

        return {
            "risk_level": "HIGH RISK",
            "reason": (
                "The machine-learning model detected "
                "a high probability of phishing indicators."
            ),
        }

    if phishing_probability >= 0.50:
        return {
            "risk_level": "SUSPICIOUS",
            "reason": (
                "The model detected some phishing indicators, "
                "but the prediction is not strong enough to classify "
                "the URL as high risk."
            ),
        }

    if legitimate_probability >= 0.75:
        return {
            "risk_level": "LOW RISK",
            "reason": (
                "The model classified the URL as likely legitimate "
                "with a strong legitimate probability."
            ),
        }

    return {
        "risk_level": "SUSPICIOUS",
        "reason": (
            "The model prediction is uncertain. "
            "Verify the website before entering sensitive information."
        ),
    }