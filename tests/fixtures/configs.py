"""Test config paths."""
import os
import pytest


@pytest.fixture
def config_dir(pytestconfig):
    """Get the dir path to configs."""
    return os.path.join(pytestconfig.rootdir, "tests/configs")


@pytest.fixture
def base_config(config_dir):
    """Get the full path to base config."""
    return os.path.join(config_dir, "base.yaml")


@pytest.fixture
def params_config(config_dir):
    """Get the full path to the supplementary params config."""
    return os.path.join(config_dir, "params.yaml")
