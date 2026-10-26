import pytest

from camie_tagger.config import ConfigError, load_settings


@pytest.mark.parametrize("threshold", [0, 0.5, 1])
def test_valid_threshold_boundaries(threshold):
    assert load_settings(threshold=threshold).threshold == threshold


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_global_thresholds(threshold):
    with pytest.raises(ConfigError):
        load_settings(threshold=threshold)


@pytest.mark.parametrize(
    "value", ['[]', '{invalid', '{"unknown":0.5}', '{"character":-1}', '{"character":NaN}']
)
def test_invalid_category_thresholds(monkeypatch, value):
    monkeypatch.setenv("CAMIE_CATEGORY_THRESHOLDS", value)
    with pytest.raises(ConfigError):
        load_settings()


def test_category_threshold_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / "config.env"
    env_file.write_text(
        'CAMIE_THRESHOLD=0.4\nCAMIE_CATEGORY_THRESHOLDS={"character":0.4}\n', encoding="utf-8"
    )
    monkeypatch.setenv("CAMIE_THRESHOLD", "0.6")
    monkeypatch.setenv("CAMIE_CATEGORY_THRESHOLDS", '{"character":0.7,"copyright":0.6}')

    settings = load_settings(env_file, threshold=0.8, category_thresholds={"character": 0.9})

    assert settings.threshold == 0.8
    assert settings.category_thresholds == {"character": 0.9, "copyright": 0.6}


def test_category_thresholds_from_dotenv(tmp_path):
    env_file = tmp_path / "config.env"
    env_file.write_text('CAMIE_CATEGORY_THRESHOLDS={"character":0.4}\n', encoding="utf-8")
    assert load_settings(env_file).category_thresholds == {"character": 0.4}


def test_cli_threshold_overrides_invalid_environment_value(monkeypatch):
    monkeypatch.setenv("CAMIE_THRESHOLD", "invalid")
    assert load_settings(threshold=0.6).threshold == 0.6


def test_cli_category_override_is_validated_after_merging(monkeypatch):
    monkeypatch.setenv("CAMIE_CATEGORY_THRESHOLDS", '{"character":"invalid","copyright":0.4}')
    assert load_settings(category_thresholds={"character": 0.6}).category_thresholds == {
        "character": 0.6, "copyright": 0.4
    }