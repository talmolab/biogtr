"""Module containing training, validation and inference logic."""

from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
from biogtr.inference.tracker import Tracker
from biogtr.inference import metrics
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.training.losses import AssoLoss
from biogtr.models.model_utils import init_optimizer, init_scheduler
from pytorch_lightning import LightningModule


class GTRRunner(LightningModule):
    """A lightning wrapper around GTR model.

    Used for training, validation and inference.
    """

    def __init__(
        self,
        model_cfg: dict = {},
        tracker_cfg: dict = {},
        loss_cfg: dict = {},
        optimizer_cfg: dict = None,
        scheduler_cfg: dict = None,
        train_metrics: list[str] = [""],
        val_metrics: list[str] = ["sw_cnt"],
        test_metrics: list[str] = ["sw_cnt"],
    ):
        """Initialize a lightning module for GTR.

        Args:
            model_cfg: hyperparameters for GlobalTrackingTransformer
            tracker_cfg: The parameters used for the tracker post-processing
            loss_cfg: hyperparameters for AssoLoss
            optimizer_cfg: hyper parameters used for optimizer.
                       Only used to overwrite `configure_optimizer`
            scheduler_cfg: hyperparameters for lr_scheduler used to overwrite `configure_optimizer
            train_metrics: a list of metrics to be calculated during training
            val_metrics: a list of metrics to be calculated during validation
            test_metrics: a list of metrics to be calculated at test time
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = GlobalTrackingTransformer(**model_cfg)
        self.loss = AssoLoss(**loss_cfg)

        self.tracker_cfg = tracker_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.train_metrics = train_metrics
        self.val_metrics = val_metrics
        self.test_metrics = test_metrics

    def forward(self, instances) -> torch.Tensor:
        """The forward pass of the lightning module.

        Args:
            instances: a list of dicts where each dict is a frame with gt data

        Returns:
            An association matrix between objects
        """
        return self.model(instances)

    def training_step(
        self, train_batch: list[dict], batch_idx: int
    ) -> dict[str, float]:
        """Method outlining the training procedure for model.

        Args:
            train_batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the train loss plus any other metrics specified
        """
        result = self._shared_eval_step(train_batch[0], self.train_metrics)
        for metric, val in result.items():
            self.log(f"train_{metric}", val, batch_size=len(train_batch[0]))
        return result

    def validation_step(
        self, val_batch: list[dict], batch_idx: int
    ) -> dict[str, float]:
        """Method outlining the val procedure for model.

        Args:
            val_batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the val loss plus any other metrics specified
        """
        result = self._shared_eval_step(val_batch[0], eval_metrics=self.val_metrics)
        for metric, val in result.items():
            self.log(f"val_{metric}", val, batch_size=len(val_batch[0]))
        return result

    def test_step(self, test_batch: list[dict], batch_idx: int) -> dict[str, float]:
        """Method outlining the test procedure for model.

        Args:
            val_batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the val loss plus any other metrics specified
        """
        result = self._shared_eval_step(test_batch[0], eval_metrics=self.test_metrics)
        for metric, val in result.items():
            self.log(f"val_{metric}", val, batch_size=len(test_batch[0]))
        return result

    def predict_step(self, batch: list[dict], batch_idx: int) -> dict:
        """Method describing inference for model.

        Computes association + assignment.

        Args:
            batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning

        Returns:
            A list of dicts where each dict is a frame containing the predicted track ids
        """
        tracker = Tracker(self.model, **self.tracker_cfg)
        instances_pred = tracker(batch[0])
        return instances_pred

    def _shared_eval_step(self, instances, eval_metrics=["sw_cnt"]):
        """Helper function for running evaluation used by train, test, and val steps.

        Args:
            instances: A list of dicts where each dict is a frame containing gt data
            eval_metrics: A list of metrics calculated and saved

        Returns:
            a dict containing the loss and any other metrics specified by `eval_metrics`
        """
        if self.model.transformer.return_embedding:
            logits, _ = self(instances)
        else:
            logits = self(instances)
        loss = self.loss(logits, instances)

        return_metrics = {"loss": loss}
        if "sw_cnt" in eval_metrics:
            tracker = Tracker(self.model, **self.tracker_cfg)
            instances_pred = tracker(instances)
            matches, indices, _ = metrics.get_matches(instances_pred)
            switches = metrics.get_switches(matches, indices)
            sw_cnt = metrics.get_switch_count(switches)
            return_metrics["sw_cnt"] = sw_cnt
        return return_metrics

    def configure_optimizers(self) -> dict:
        """Get optimizers and schedulers for training.

        Is overridden by config but defaults to Adam + ReduceLROnPlateau.

        Returns:
            an optimizer config dict containing the optimizer, scheduler, and scheduler params
        """
        # todo: init from config
        if self.optimizer_cfg is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, betas=(0.9, 0.999))
        else:
            optimizer = init_optimizer(self.parameters(), self.optimizer_cfg)

        if self.scheduler_cfg is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", 0.5, 10
            )
        else:
            scheduler = init_scheduler(optimizer, self.scheduler_cfg)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
                "interval": "epoch",
                "frequency": 10,
            },
        }
