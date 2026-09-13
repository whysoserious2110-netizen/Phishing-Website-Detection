from urllib.parse import urlsplit
import ipaddress
import tldextract


def analyze_domain(url):
    url = str(url).strip()

    if not url:
        return {
            "hostname": "",
            "main_domain": "",
            "subdomain_count": 0,
            "is_ip_address": False
        }

    if "://" not in url:
        url = "http://" + url

    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return {
            "hostname": "",
            "main_domain": "",
            "subdomain_count": 0,
            "is_ip_address": False
        }

    try:
        ipaddress.ip_address(hostname)
        is_ip_address = True
    except ValueError:
        is_ip_address = False

    if is_ip_address:
        return {
            "hostname": hostname,
            "main_domain": hostname,
            "subdomain_count": 0,
            "is_ip_address": True
        }

    extracted = tldextract.extract(hostname)

    if extracted.domain and extracted.suffix:
        main_domain = extracted.domain + "." + extracted.suffix
    else:
        main_domain = hostname

    if extracted.subdomain:
        subdomain_count = len(extracted.subdomain.split("."))
    else:
        subdomain_count = 0

    return {
        "hostname": hostname,
        "main_domain": main_domain,
        "subdomain_count": subdomain_count,
        "is_ip_address": False
    }