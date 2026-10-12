"""Tests for configuration management and environment detection."""

from pathlib import Path
import tempfile
import yaml

from src.config import ProjectConfig, is_kaggle_environment, PathsConfig


def test_default_config_initialization():
    config = ProjectConfig()
    assert config.project_name == "spoken-grammar-scoring"
    assert config.audio.target_sample_rate == 16000
    assert config.audio.mono is True
    assert config.validation.n_splits == 5
    assert config.validation.random_seed == 42


def test_paths_derived_properties():
    paths = PathsConfig(data_dir=Path("/test_dir"))
    assert paths.train_csv_path == Path("/test_dir/train.csv")
    assert paths.test_csv_path == Path("/test_dir/test.csv")
    assert paths.sample_submission_path == Path("/test_dir/sample_submission.csv")


def test_config_yaml_loading():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        yaml_content = {
            "project_name": "custom-grammar",
            "random_seed": 123,
            "audio": {"target_sample_rate": 22050, "mono": False},
            "validation": {"n_splits": 10},
        }
        cfg_file = tmp_p / "custom.yaml"
        with open(cfg_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        loaded_cfg = ProjectConfig.load(config_path=cfg_file, auto_kaggle=False)
        assert loaded_cfg.project_name == "custom-grammar"
        assert loaded_cfg.random_seed == 123
        assert loaded_cfg.audio.target_sample_rate == 22050
        assert loaded_cfg.audio.mono is False
        assert loaded_cfg.validation.n_splits == 10


def test_config_to_dict_serialization():
    config = ProjectConfig()
    d = config.to_dict()
    assert isinstance(d, dict)
    assert d["project_name"] == "spoken-grammar-scoring"
    assert isinstance(d["paths"]["data_dir"], str)
