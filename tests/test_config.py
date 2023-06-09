"""Tests for `config.py`"""

from omegaconf import OmegaConf
from biogtr.config import Config
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.models.gtr_runner import GTRRunner

import torch


def test_init(base_config, params_config):
    """Test the `__init__` function of `Config`.

    Load + merge

    Args:
        base_config: the initial config params
        params_config: the config params to override
    """
    base_cfg = OmegaConf.load(base_config)
    base_cfg["params_config"] = params_config
    cfg = Config(base_cfg)

    assert cfg.cfg.model.num_encoder_layers == 2
    assert cfg.cfg.dataset.train_dataset.clip_length == 32
    assert cfg.cfg.dataset.train_dataset.padding == 5
    assert "val_dataset" in cfg.cfg.dataset


def test_setter(base_config):
    """Test the setter function of `Config`.

    set hyperparameters

    Args:
        base_config: the initial config params
    """
    base_cfg = OmegaConf.load(base_config)
    cfg = Config(base_cfg)
    hparams = {
        "model.num_encoder_layers": 1,
        "logging.name": "test_config",
        "dataset.train_dataset.chunk": False,
    }

    assert cfg.set_hparams(hparams)
    assert cfg.cfg.model.num_encoder_layers == 1
    assert cfg.cfg.tracker.window_size == 8

    hparams = {"test_config": -1}

    assert cfg.set_hparams(hparams)
    assert "test_config" in cfg.cfg
    assert cfg.cfg.test_config == -1


def test_getters(base_config):
    """Test each getter function in the config class.

    Args:
        base_config: the config params to override
    """
    base_cfg = OmegaConf.load(base_config)
    cfg = Config(base_cfg)

    model = cfg.get_model()
    assert isinstance(model, GlobalTrackingTransformer)
    assert model.transformer.d_model == 512

    tracker_cfg = cfg.get_tracker_cfg()
    assert set(
        [
            "window_size",
            "use_vis_feats",
            "overlap_thresh",
            "mult_thresh",
            "decay_time",
            "iou",
            "max_center_dist",
        ]
    ) == set(tracker_cfg.keys())
    assert tracker_cfg["window_size"] == 8

    gtr_runner = cfg.get_gtr_runner()
    assert isinstance(gtr_runner, GTRRunner)
    assert gtr_runner.model.transformer.d_model == 512

    ds = cfg.get_dataset("train")
    assert ds.clip_length == 4
    ds = cfg.get_dataset("val")
    assert ds.clip_length == 8
    ds = cfg.get_dataset("test")
    assert ds.clip_length == 16

    optim = cfg.get_optimizer(model.parameters())
    assert isinstance(optim, torch.optim.Adam)

    scheduler = cfg.get_scheduler(optim)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
