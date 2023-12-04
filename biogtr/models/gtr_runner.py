"""Module containing training, validation and inference logic."""
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
        metrics: dict[str, list[str]] = {
            "train": ["num_switches"],
            "val": ["num_switches"],
            "test": ["num_switches"],
        },
        persistent_tracking: dict[str, bool] = {
            "train": False,
            "val": True,
            "test": True,
        },
    ):
        """Initialize a lightning module for GTR.

        Args:
            model_cfg: hyperparameters for GlobalTrackingTransformer
            tracker_cfg: The parameters used for the tracker post-processing
            loss_cfg: hyperparameters for AssoLoss
            optimizer_cfg: hyper parameters used for optimizer.
                       Only used to overwrite `configure_optimizer`
            scheduler_cfg: hyperparameters for lr_scheduler used to overwrite `configure_optimizer
            metrics: a dict containing the metrics to be computed during train, val, and test.
            persistent_tracking: a dict containing whether to use persistent tracking during train, val and test inference.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = GlobalTrackingTransformer(**model_cfg)
        self.loss = AssoLoss(**loss_cfg)
        self.tracker = Tracker(**tracker_cfg)

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.metrics = metrics
        self.persistent_tracking = persistent_tracking

    def forward(self, instances) -> torch.Tensor:
        """Execute forward pass of the lightning module.

        Args:
            instances: a list of dicts where each dict is a frame with gt data

        Returns:
            An association matrix between objects
        """
        if sum([frame.num_detected for frame in instances]) > 0:
            asso_preds, _ = self.model(instances)
            return asso_preds
        return None

    def training_step(
        self, train_batch: list[dict], batch_idx: int
    ) -> dict[str, float]:
        """Execute single training step for model.

        Args:
            train_batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the train loss plus any other metrics specified
        """
        result = self._shared_eval_step(train_batch[0], mode="train")
        self.log_metrics(result, "train")

        return result

    def validation_step(
        self, val_batch: list[dict], batch_idx: int
    ) -> dict[str, float]:
        """Execute single val step for model.

        Args:
            val_batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the val loss plus any other metrics specified
        """
        result = self._shared_eval_step(val_batch[0], mode="val")
        self.log_metrics(result, "val")

        return result

    def test_step(self, test_batch: list[dict], batch_idx: int) -> dict[str, float]:
        """Execute single test step for model.

        Args:
            val_batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning

        Returns:
            A dict containing the val loss plus any other metrics specified
        """
        result = self._shared_eval_step(test_batch[0], mode="test")
        self.log_metrics(result, "test")

        return result

    def predict_step(self, batch: list[dict], batch_idx: int) -> dict:
        """Run inference for model.

        Computes association + assignment.

        Args:
            batch: A single batch from the dataset which is a list of dicts
            with length `clip_length` where each dict is a frame
            batch_idx: the batch number used by lightning

        Returns:
            A list of dicts where each dict is a frame containing the predicted track ids
        """
        self.tracker.persistent_tracking = True
        instances_pred = self.tracker(self.model, batch[0])
        return instances_pred

    def _shared_eval_step(self, instances, mode):
        """Run evaluation used by train, test, and val steps.

        Args:
            instances: A list of dicts where each dict is a frame containing gt data
            mode: which metrics to compute and whether to use persistent tracking or not

        Returns:
            a dict containing the loss and any other metrics specified by `eval_metrics`
        """
        try:
            eval_metrics = self.metrics[mode]
            persistent_tracking = self.persistent_tracking[mode]

            logits = self(instances)

            if not logits:
                return None

            loss = self.loss(logits, instances)

            return_metrics = {"loss": loss}
            if eval_metrics is not None and len(eval_metrics) > 0:
                self.tracker.persistent_tracking = persistent_tracking
                instances_pred = self.tracker(self.model, instances)
                instances_mm = metrics.to_track_eval(instances_pred)
                clearmot = metrics.get_pymotmetrics(instances_mm, eval_metrics)
                return_metrics.update(clearmot.to_dict())
        except Exception as e:
            print(
                f"Failed on frame {instances[0].frame_id} of video {instances[0].video_id}"
            )
            raise (e)
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
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 10,
            },
        }

    def log_metrics(self, result: dict, mode: str) -> None:
        """Log metrics computed during evaluation.

        Args:
            result: A dict containing metrics to be logged.
            mode: One of {'train', 'test' or 'val'}. Used as prefix while logging.
        """
        if result:
            for metric, val in result.items():
                self.log(f"{mode}_{metric}", val, on_step=True, on_epoch=True)
