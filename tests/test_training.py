"""Test training logic."""
import os
import pytest
import torch
from biogtr.training.losses import AssoLoss
from biogtr.models.gtr_runner import GTRRunner
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from omegaconf import OmegaConf, DictConfig
from biogtr.config import Config
from biogtr.training.train import main

# todo: add named tensor tests


def test_asso_loss():
    """Test asso loss."""
    num_frames = 5
    num_detected = 20
    img_shape = (1, 128, 128)

    instances = []

    for i in range(num_frames):
        instances.append(
            {
                "img_shape": torch.tensor(img_shape),
                "num_detected": torch.tensor([num_detected]),
                "gt_track_ids": torch.arange(num_detected),
                "bboxes": torch.rand(size=(num_detected, 4)),
            }
        )

    asso_loss = AssoLoss(neg_unmatched=True, asso_weight=10.0)

    asso_preds = torch.rand(size=(1, 100, 100))

    loss = asso_loss(asso_preds, instances)

    assert len(loss.size()) == 0
    assert type(loss.item()) == float


def test_basic_gtr_runner():
    """Test basic GTR Runner."""
    feats = 128
    num_frames = 2
    num_detected = 3
    img_shape = (1, 128, 128)
    n_batches = 2
    instances = []
    train_ds = []
    epochs = 2

    for i in range(n_batches):
        for j in range(num_frames):
            instances.append(
                {
                    "video_id": torch.tensor(0),
                    "frame_id": torch.tensor(j),
                    "img_shape": torch.tensor(img_shape),
                    "num_detected": torch.tensor([num_detected]),
                    "crops": torch.rand(size=(num_detected, 1, 64, 64)),
                    "bboxes": torch.rand(size=(num_detected, 4)),
                    "gt_track_ids": torch.arange(num_detected),
                    "pred_track_ids": torch.tensor([-1] * num_detected),
                }
            )
        train_ds.append([instances])

    gtr_runner = GTRRunner()

    optim_scheduler = gtr_runner.configure_optimizers()

    assert isinstance(optim_scheduler["optimizer"], torch.optim.Adam)
    assert isinstance(
        optim_scheduler["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    optim_cfg = {"name": "SGD", "lr": 1e-3}
    scheduler_cfg = {"name": "CosineAnnealingLR", "T_max": 1}

    gtr_runner = GTRRunner(optimizer_cfg=optim_cfg, scheduler_cfg=scheduler_cfg)
    optim_scheduler = gtr_runner.configure_optimizers()

    assert isinstance(optim_scheduler["optimizer"], torch.optim.SGD)
    assert isinstance(
        optim_scheduler["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.CosineAnnealingLR,
    )

    for epoch in range(epochs):
        for i, batch in enumerate(train_ds):
            assert gtr_runner.model.training
            metrics = gtr_runner.training_step(batch, i)
            assert "loss" in metrics and "sw_cnt" not in metrics
            assert metrics["loss"].requires_grad

        for j, batch in enumerate(train_ds):
            gtr_runner.eval()
            with torch.no_grad():
                metrics = gtr_runner.validation_step(batch, j)
            assert "loss" in metrics and "sw_cnt" in metrics
            assert not metrics["loss"].requires_grad

        gtr_runner.train()

    for k, batch in enumerate(train_ds):
        gtr_runner.eval()
        with torch.no_grad():
            metrics = gtr_runner.test_step(batch, k)
        assert "loss" in metrics and "sw_cnt" in metrics
        assert not metrics["loss"].requires_grad


# temp fix for actions test, still need to debug
@pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") == "true",
    reason="Silent fail on GitHub Actions",
)
def test_config_gtr_runner(base_config, params_config, two_flies):
    """Test config GTR Runner."""
    base_cfg = OmegaConf.load(base_config)
    base_cfg["params_config"] = params_config
    cfg = Config(base_cfg)

    hparams = {
        "dataset.clip_length": 8,
        "trainer.min_epochs": 1,
    }

    cfg.set_hparams(hparams)

    main(cfg.cfg)
