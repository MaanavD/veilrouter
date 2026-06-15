from veilrouter.pii.detector import RegexPiiDetector, Span
from veilrouter.pii.placeholders import PlaceholderFactory, category_label, is_placeholder
from veilrouter.pii.redactor import Redactor
from veilrouter.pii.restorer import StreamRestorer, restore_text


def test_regex_detector_finds_common_pii_and_ignores_invalid_credit_cards():
    text = (
        "Send a note to Ada Lovelace at ada@example.com, SSN 123-45-6789, phone +1 (425) 555-0100, "
        "card 4111 1111 1111 1111, invalid card 4111 1111 1111 1112."
    )

    spans = RegexPiiDetector().detect(text)
    by_category = {span.category: span.text for span in spans}

    assert by_category["identity.person_name"] == "Ada Lovelace"
    assert by_category["contact.email"] == "ada@example.com"
    assert by_category["identity.ssn"] == "123-45-6789"
    assert by_category["contact.phone"] == "+1 (425) 555-0100"
    assert by_category["financial.credit_card"] == "4111 1111 1111 1111"
    assert "4111 1111 1111 1112" not in by_category.values()


def test_regex_detector_redacts_contextual_person_names_without_contact_details():
    spans = RegexPiiDetector().detect("Please brief customer Jane Doe about the renewal.")

    assert [span.text for span in spans if span.category == "identity.person_name"] == ["Jane Doe"]


class FixedDetector:
    def detect(self, text: str):
        target = "Ada Lovelace"
        start = text.find(target)
        if start < 0:
            return []
        return [Span("identity.person_name", start, start + len(target), target, 0.99)]


def test_redactor_deep_copies_messages_and_redacts_nested_text():
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello Ada Lovelace"}]}]
    redactor = Redactor(FixedDetector(), regex_backstop=False)

    result = redactor.redact_messages(messages)

    assert messages[0]["content"][0]["text"] == "Hello Ada Lovelace"
    assert result.messages[0]["content"][0]["text"] == "Hello [PERSON_NAME_1]"
    assert result.placeholder_to_original == {"[PERSON_NAME_1]": "Ada Lovelace"}
    assert result.original_to_placeholder == {"Ada Lovelace": "[PERSON_NAME_1]"}
    assert result.redaction_count == 1
    assert result.categories == {"PERSON_NAME": 1}


def test_redactor_reuses_placeholder_for_repeated_originals():
    messages = [{"role": "user", "content": "Email ada@example.com and ada@example.com again"}]

    result = Redactor(regex_backstop=True).redact_messages(messages)

    assert result.messages[0]["content"] == "Email [EMAIL_1] and [EMAIL_1] again"
    assert result.redaction_count == 1
    assert result.categories == {"EMAIL": 1}


def test_redactor_redacts_person_name_email_and_phone_together():
    messages = [{"role": "user", "content": "Send a summary to Jane Doe at jane@example.com and call +1 (425) 555-0199."}]

    result = Redactor(regex_backstop=True).redact_messages(messages)

    assert result.messages[0]["content"] == "Send a summary to [PERSON_NAME_1] at [EMAIL_1] and call [PHONE_1]."
    assert result.placeholder_to_original["[PERSON_NAME_1]"] == "Jane Doe"
    assert result.placeholder_to_original["[EMAIL_1]"] == "jane@example.com"
    assert result.placeholder_to_original["[PHONE_1]"] == "+1 (425) 555-0199"
    assert result.categories == {"PERSON_NAME": 1, "EMAIL": 1, "PHONE": 1}


def test_restore_text_replaces_known_placeholders_and_leaves_unknown_values():
    restored = restore_text("Known [EMAIL_1], unknown [EMAIL_2]", {"[EMAIL_1]": "ada@example.com"})

    assert restored == "Known ada@example.com, unknown [EMAIL_2]"


def test_stream_restorer_handles_placeholders_split_across_chunks():
    restorer = StreamRestorer({"[EMAIL_1]": "ada@example.com"})

    first = restorer.feed("Hello [EMA")
    second = restorer.feed("IL_1]!")
    tail = restorer.finish()

    assert first == "Hello "
    assert second == "ada@example.com!"
    assert tail == ""


def test_placeholder_helpers_normalize_categories_and_validate_shape():
    factory = PlaceholderFactory()

    placeholder = factory.create("contact.email")

    assert placeholder == "[EMAIL_1]"
    assert category_label("identity person-name") == "PERSON_NAME"
    assert is_placeholder("[EMAIL_1]")
    assert not is_placeholder("[EMAIL_0]")
