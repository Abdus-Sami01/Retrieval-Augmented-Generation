import re

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")


def redact_pii(text: str) -> str:
    """Regex-based redaction for email, US-style SSN, and US-style phone numbers.
    SSN is redacted before phone since it's the more specific pattern - an SSN
    (XXX-XX-XXXX) would not match the phone regex, but redacting order-independence
    isn't guaranteed for all pattern pairs in general, so specific-first is the rule."""
    text = _SSN_RE.sub("[SSN_REDACTED]", text)
    text = _EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE_REDACTED]", text)
    return text
