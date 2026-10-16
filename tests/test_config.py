from pathlib import Path

import pytest

from config import ConfigError, ExperimentConfig, config_from_dict, config_to_dict, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load_and_validate(path):
    cfg = load_config(path)
    assert cfg.num_classes == 91
    assert cfg.data.sequence_length == 10


def test_overrides_and_yaml_float_strings():
    cfg = load_config(CONFIG_DIR / "debug.yaml", ["train.lr=1e-3", "augment.mode=none", "train.max_steps=null"])
    assert cfg.train.lr == pytest.approx(1e-3) and isinstance(cfg.train.lr, float)
    assert cfg.augment.mode == "none" and cfg.train.max_steps is None


@pytest.mark.parametrize(
    "override",
    ["train.learning_rate=0.1", "augment.mode=flip", "train.batch_size=abc", "train.batch_size=true", "model.nhead=3", "data.image_probability=1.5"],
)
def test_invalid_values_raise(override):
    with pytest.raises(ConfigError):
        load_config(CONFIG_DIR / "debug.yaml", [override])


def test_bf16_requires_cuda():
    with pytest.raises(ConfigError):
        load_config(CONFIG_DIR / "debug.yaml", ["train.device=mps", "train.amp_dtype=bf16"])


def test_dict_round_trip():
    cfg = ExperimentConfig()
    assert config_from_dict(config_to_dict(cfg)) == cfg
