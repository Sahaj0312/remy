"""The rule layer of the reply classifier runs before any model call: empty -> no_answer, a help word ->
needs_help. Anything else goes to the triage backend, which is faked here so no API is touched."""
from remy.checkin import classify, question_text


class FakeBackend:
    def __init__(self, status):
        self.status = status
        self.calls = []

    def classify_reply(self, transcript, question):
        self.calls.append((transcript, question))
        return {"status": self.status, "quote": transcript, "reason": "fake", "rule": False, "latency_s": 0.0, "cost_usd": 0.0}


def test_question_text_fills_in_the_name(cfg):
    assert question_text(cfg["checkin"]) == "Nani, are you okay?"


def test_empty_transcript_is_no_answer_without_a_model_call(cfg):
    be = FakeBackend("okay")
    r = classify("   ", cfg["checkin"], "q", backend=be)
    assert r["status"] == "no_answer" and r["rule"] is True and be.calls == []


def test_help_words_force_needs_help_in_any_case_and_language(cfg):
    be = FakeBackend("okay")
    for text in ("Please HELP me", "bachao", "madad karo", "I can't get up", "my hip hurts"):
        r = classify(text, cfg["checkin"], "q", backend=be)
        assert r["status"] == "needs_help", text
        assert r["rule"] is True
    assert be.calls == []


def test_other_speech_goes_to_the_backend(cfg):
    be = FakeBackend("unclear")
    r = classify("what time is it", cfg["checkin"], "Nani, are you okay?", backend=be)
    assert r["status"] == "unclear" and r["rule"] is False
    assert be.calls == [("what time is it", "Nani, are you okay?")]


def test_okay_wording_is_the_backend_call_not_a_keyword(cfg):
    be = FakeBackend("okay")
    r = classify("no no I'm fine, just slipped", cfg["checkin"], "q", backend=be)
    assert r["status"] == "okay" and len(be.calls) == 1
