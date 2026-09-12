from urllib.parse import urlparse
import re


def extract_url_features_no_https(url):

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    domain = parsed.hostname or ""

    url_length = len(url)
    domain_length = len(domain)

    is_domain_ip = 0

    parts = domain.split(".")

    if len(parts) == 4:
        if all(
            part.isdigit() and 0 <= int(part) <= 255
            for part in parts
        ):
            is_domain_ip = 1

    if "." in domain:
        tld = domain.rsplit(".", 1)[-1]
    else:
        tld = ""

    tld_length = len(tld)

    domain_parts = domain.split(".")

    if len(domain_parts) > 2:
        no_of_subdomain = len(domain_parts) - 2
    else:
        no_of_subdomain = 0

    encoded_parts = re.findall(
        r"%[0-9A-Fa-f]{2}",
        url
    )

    no_of_obfuscated_char = len(encoded_parts)

    has_obfuscation = (
        1 if no_of_obfuscated_char > 0 else 0
    )

    if url_length > 0:
        obfuscation_ratio = (
            no_of_obfuscated_char / url_length
        )
    else:
        obfuscation_ratio = 0

    no_of_letters = sum(
        char.isalpha()
        for char in url
    )

    if url_length > 0:
        letter_ratio = (
            no_of_letters / url_length
        )
    else:
        letter_ratio = 0

    no_of_digits = sum(
        char.isdigit()
        for char in url
    )

    if url_length > 0:
        digit_ratio = (
            no_of_digits / url_length
        )
    else:
        digit_ratio = 0

    no_of_equals = url.count("=")
    no_of_qmark = url.count("?")
    no_of_ampersand = url.count("&")

    no_of_other_special_chars = sum(
        not char.isalnum()
        and char not in ["=", "?", "&"]
        for char in url
    )

    if url_length > 0:
        special_char_ratio = (
            no_of_other_special_chars / url_length
        )
    else:
        special_char_ratio = 0

    features = [
        url_length,
        domain_length,
        is_domain_ip,
        tld_length,
        no_of_subdomain,
        has_obfuscation,
        no_of_obfuscated_char,
        obfuscation_ratio,
        no_of_letters,
        letter_ratio,
        no_of_digits,
        digit_ratio,
        no_of_equals,
        no_of_qmark,
        no_of_ampersand,
        no_of_other_special_chars,
        special_char_ratio
    ]

    return features