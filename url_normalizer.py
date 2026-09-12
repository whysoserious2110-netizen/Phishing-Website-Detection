from urllib.parse import urlsplit, urlunsplit


def normalize_url(url):
    """
    Normalize only the root path of a URL.

    Examples:
        https://facebook.com/      -> https://facebook.com
        https://facebook.com       -> https://facebook.com
        https://facebook.com/login -> unchanged
        https://facebook.com/login/ -> unchanged
        https://facebook.com/?a=1 -> unchanged
    """

    url = str(url).strip()

    if not url:
        return url

    # Add a temporary scheme so urlsplit can correctly identify the hostname.
    has_scheme = "://" in url

    if not has_scheme:
        parsed = urlsplit("http://" + url)
        prefix_added = True
    else:
        parsed = urlsplit(url)
        prefix_added = False

    # Remove only the root "/" when there is no query or fragment.
    if parsed.path == "/" and not parsed.query and not parsed.fragment:
        parsed = parsed._replace(path="")

    normalized = urlunsplit(parsed)

    # Restore the original scheme-less form if the user entered one.
    if prefix_added and normalized.startswith("http://"):
        normalized = normalized[len("http://"):]

    return normalized