from edge.config import EdgeConfig, load_config


def test_default_config_is_valid_mock_mode():
    config = EdgeConfig()
    assert config.hardware.mode == "mock"
    assert config.duty_cycle.window_duration_s > 0
    assert config.duty_cycle.window_interval_minutes > 0


def test_load_config_missing_file_returns_defaults(tmp_path):
    config = load_config(tmp_path / "does_not_exist.yaml")
    assert config == EdgeConfig()


def test_load_config_applies_yaml_overrides(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
duty_cycle:
  window_duration_s: 3.0
  window_interval_minutes: 15.0
hardware:
  mode: real
  i2c:
    bus_number: 2
"""
    )
    config = load_config(yaml_path)

    assert config.duty_cycle.window_duration_s == 3.0
    assert config.duty_cycle.window_interval_minutes == 15.0
    assert config.hardware.mode == "real"
    assert config.hardware.i2c.bus_number == 2
    # untouched nested fields keep their defaults
    assert config.hardware.i2c.imu_address == "0x68"


def test_load_config_rejects_unknown_key(tmp_path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("not_a_real_field: 123\n")

    try:
        load_config(yaml_path)
        assert False, "expected ValueError for unknown config key"
    except ValueError:
        pass


def test_default_repo_config_yaml_loads_cleanly():
    # edge/config.yaml itself (the shipped default file) must parse and
    # merge without error -- this is what edge/main.py loads with no args.
    config = load_config()
    assert config.hardware.mode == "mock"
