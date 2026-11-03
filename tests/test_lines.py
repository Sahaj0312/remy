from pathlib import Path

from remy.checkin import question_text
from remy.checkin.tts import LINE_KEYS, available_keys, line_text, prompt_path


def test_every_line_has_text_with_the_name_filled_in(cfg):
    ck = cfg["checkin"]
    assert question_text(ck) == "Are you okay? Can you hear me?"
    for key in LINE_KEYS:
        t = line_text(ck, key)
        assert "{name}" not in t and t.strip()
    assert available_keys(ck) == list(LINE_KEYS)


def test_old_prompts_config_still_gives_the_question():
    ck = {"language": "en", "name": "Nani", "prompts": {"en": "{name}, are you okay?"}}
    assert line_text(ck, "ask") == "Nani, are you okay?"
    assert available_keys(ck) == ["ask"]


def test_prompt_paths_are_per_line_and_language():
    assert prompt_path("ask_again", "en").name == "ask_again_en.wav"
    assert prompt_path("ack_okay", "hi").name == "ack_okay_hi.wav"
