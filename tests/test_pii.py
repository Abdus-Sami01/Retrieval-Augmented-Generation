from rag.pii import redact_pii


def test_redacts_email():
    result = redact_pii("Contact me at jane.doe@example.com for details.")
    assert "jane.doe@example.com" not in result
    assert "[EMAIL_REDACTED]" in result


def test_redacts_ssn():
    result = redact_pii("SSN on file: 123-45-6789.")
    assert "123-45-6789" not in result
    assert "[SSN_REDACTED]" in result


def test_redacts_phone_with_dashes():
    result = redact_pii("Call us at 555-123-4567 anytime.")
    assert "555-123-4567" not in result
    assert "[PHONE_REDACTED]" in result


def test_redacts_phone_with_parens_and_country_code():
    result = redact_pii("Reach me at +1 (555) 123-4567 during business hours.")
    assert "(555) 123-4567" not in result
    assert "[PHONE_REDACTED]" in result


def test_leaves_unrelated_text_unchanged():
    text = "The quarterly report shows a 12% increase in revenue."
    assert redact_pii(text) == text


def test_ssn_not_double_redacted_as_phone():
    result = redact_pii("SSN: 123-45-6789")
    assert result.count("REDACTED") == 1


def test_redacts_multiple_pii_types_in_one_text():
    text = "Email jane@example.com or call 555-123-4567. SSN 987-65-4321."
    result = redact_pii(text)
    assert "jane@example.com" not in result
    assert "555-123-4567" not in result
    assert "987-65-4321" not in result
    assert "[EMAIL_REDACTED]" in result
    assert "[PHONE_REDACTED]" in result
    assert "[SSN_REDACTED]" in result
