import pytest

from cc_model_router import DEFAULT_WEB_SEARCH_MODEL, load_config
from conftest import MODEL


def write_config(tmp_path, top=""):
    token = tmp_path / "api-key"
    token.write_text("backend-token\n")
    config = tmp_path / "config.toml"
    config.write_text(f'{top}\n[models."{MODEL}"]\nbase_url = "http://127.0.0.1:9931"\ntoken_file = "{token}"\n')
    return config


def test_web_search_model_naming_an_open_weight_model_is_refused(tmp_path):
    with pytest.raises(SystemExit) as exc:
        load_config(write_config(tmp_path, f'web_search_model = "{MODEL}"'))

    assert "web_search_model" in str(exc.value.code) and MODEL in str(exc.value.code)


def test_web_search_model_is_a_claude_model_by_default_or_from_the_config(tmp_path):
    assert load_config(write_config(tmp_path))["web_search_model"] == DEFAULT_WEB_SEARCH_MODEL
    assert load_config(write_config(tmp_path, 'web_search_model = "claude-sonnet-5-5"'))[
        "web_search_model"] == "claude-sonnet-5-5"
