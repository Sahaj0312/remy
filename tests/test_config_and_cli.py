from remy import cli


def test_example_config_has_every_section_the_pipeline_reads(cfg):
    for key in ("yolo", "tracking", "score_v2", "trigger", "spark", "triage", "audio", "checkin", "monitor", "alerts"):
        assert key in cfg, key
    ck = cfg["checkin"]
    assert set(ck["prompts"]) == set(ck["voices"])
    assert ck["record_s"] > 0 and ck["help_words"]
    assert cfg["alerts"]["family"][0]["number"] == "+10000000000"      # placeholder only; real numbers live in family.json
    assert cfg["triage"]["backend"] == "spark"


def test_cli_lists_its_commands_and_rejects_unknown_ones(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for c in ("run", "setup-zones", "test-checkin", "test-alert", "costs"):
        assert c in out
    assert cli.main(["nope"]) == 2
