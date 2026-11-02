import pytest

from remy import triage
from remy.triage import BACKENDS, TriageBackend, backend_name, get_backend


def test_both_backends_build_without_network(cfg, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key")
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    for name in BACKENDS:
        be = get_backend({**cfg, "triage": {"backend": name}}, timeout=5)
        assert isinstance(be, TriageBackend) and be.name == name and be.label
        assert callable(be.triage_fall) and callable(be.classify_reply)


def test_roles_pick_different_backends(cfg, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    c = {**cfg, "triage": {"backend": "spark_open", "reply_backend": "spark"}}
    assert backend_name(c, "triage") == "spark_open" and backend_name(c, "reply") == "spark"
    c = {**cfg, "triage": {"backend": "spark_open"}}
    assert backend_name(c, "reply") == "spark_open"          # reply follows triage when not set


def test_missing_partner_key_falls_back_to_spark(cfg, monkeypatch):
    monkeypatch.setattr(triage, "env_get", lambda *a, **k: None)
    assert backend_name({**cfg, "triage": {"backend": "spark_open"}}, "triage") == "spark"


def test_unknown_backend_is_a_clear_error(cfg):
    with pytest.raises(SystemExit):
        get_backend({**cfg, "triage": {"backend": "glimmer"}})


def test_spark_open_defaults_fill_a_missing_section(cfg, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    c = {k: v for k, v in cfg.items() if k != "spark_open"}
    be = get_backend({**c, "triage": {"backend": "spark_open"}}, timeout=5)
    assert be.cfg["model"].endswith("muse-spark-open") and be.cfg["reasoning_effort"] == "low"
