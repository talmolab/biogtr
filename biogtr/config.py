# to implement - config class that handles getters/setters
"""Data structures for handling config parsing."""
from biogtr.datasets.microscopy_dataset import MicroscopyDataset
from biogtr.datasets.sleap_dataset import SleapDataset
from biogtr.datasets.cell_tracking_dataset import CellTrackingDataset
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.models.gtr_runner import GTRRunner
from biogtr.models.model_utils import init_optimizer, init_scheduler, init_logger
from biogtr.training.losses import AssoLoss
from omegaconf import DictConfig, OmegaConf
from pprint import pprint
from typing import Union, Iterable
import pytorch_lightning as pl
import torch


class Config:
    """Class handling loading components based on config params."""

    def __init__(self, cfg: DictConfig):
        """Initialize the class with config from hydra/omega conf.

        First uses `base_param` file then overwrites with specific `params_config`.

        Args:
            cfg: The `DictConfig` containing all the hyperparameters needed for
                training/evaluation
        """
        base_cfg = cfg
        print(f"Base Config: {cfg}")

        if "params_config" in cfg:
            # merge configs
            params_config = OmegaConf.load(cfg.params_config)
            pprint(f"Overwriting base config with {params_config}")
            self.cfg = OmegaConf.merge(base_cfg, params_config)
        else:
            # just use base config
            self.cfg = base_cfg

    def __repr__(self):
        """Object representation of config class."""
        return f"Config({self.cfg})"

    def __str__(self):
        """String representation of config class."""
        return f"Config({self.cfg})"

    def set_hparams(self, hparams: dict) -> bool:
        """Setter function for overwriting specific hparams.

        Useful for changing 1 or 2 hyperparameters such as dataset.

        Args:
            hparams: A dict containing the hyperparameter to be overwritten and
                the value to be changed

        Returns:
            `True` if config is successfully updated, `False` otherwise
        """
        if hparams == {} or hparams is None:
            print("Nothing to update!")
            return False
        for hparam, val in hparams.items():
            try:
                OmegaConf.update(self.cfg, hparam, val)
            except Exception as e:
                print(f"Failed to update {hparam} to {val} due to {e}")
                return False
        return True

    def get_model(self) -> GlobalTrackingTransformer:
        """Getter for gtr model.

        Returns:
            A global tracking transformer with parameters indicated by cfg
        """
        model_params = self.cfg.model
        return GlobalTrackingTransformer(**model_params)

    def get_tracker_cfg(self) -> dict:
        """Getter for tracker config params.

        Returns:
            A dict containing the init params for `Tracker`.
        """
        tracker_params = self.cfg.tracker
        tracker_cfg = {}
        for key, val in tracker_params.items():
            tracker_cfg[key] = val
        return tracker_cfg

    def get_gtr_runner(self):
        """Get lightning module for training, validation, and inference."""
        
        tracker_params = self.cfg.tracker
        optimizer_params = self.cfg.optimizer
        scheduler_params = self.cfg.scheduler
        loss_params = self.cfg.loss
        gtr_runner_params = self.cfg.runner
        
        if self.cfg.model.ckpt_path is not None and self.cfg.model.ckpt_path != "":
            model = GTRRunner.load_from_checkpoint(self.cfg.model.ckpt_path,
                                                   tracker_cfg = tracker_params,
                                                   train_metrics=self.cfg.runner.train_metrics,
                                                  val_metrics=self.cfg.runner.val_metrics,
                                                  test_metrics=self.cfg.runner.test_metrics)
            
        else:
            model_params = self.cfg.model
            model = GTRRunner(
                model_params,
                tracker_params,
                loss_params,
                optimizer_params,
                scheduler_params,
                **gtr_runner_params,
            )

        return model

    def get_dataset(
        self, mode: str
    ) -> Union[SleapDataset, MicroscopyDataset, CellTrackingDataset]:
        """Getter for datasets.

        Args:
            mode: [None, "train", "test", "val"]. Indicates whether to use
                train, val, or test params for dataset

        Returns:
            Either a `SleapDataset` or `MicroscopyDataset` with params indicated by cfg
        """
        if mode.lower() == "train":
            dataset_params = self.cfg.dataset.train_dataset
        elif mode.lower() == "val":
            dataset_params = self.cfg.dataset.val_dataset
        elif mode.lower() == "test":
            dataset_params = self.cfg.dataset.test_dataset
        else:
            raise ValueError(
                "`mode` must be one of ['train', 'val','test'], not '{mode}'"
            )

        # todo: handle this better
        if "slp_files" in dataset_params:
            return SleapDataset(**dataset_params)
        elif "tracks" in dataset_params or "source" in dataset_params:
            return MicroscopyDataset(**dataset_params)
        elif "raw_images" in dataset_params:
            return CellTrackingDataset(**dataset_params)
        else:
            raise ValueError(
                "Could not resolve dataset type from Config! Please include \
                either `slp_files` or `tracks`/`source`"
            )

    def get_dataloader(
        self,
        dataset: Union[SleapDataset, MicroscopyDataset, CellTrackingDataset],
        mode: str,
    ) -> torch.utils.data.DataLoader:
        """Getter for dataloader.

        Args:
            dataset: the Sleap or Microscopy Dataset used to initialize the dataloader
            mode: either ["train", "val", or "test"] indicates which dataset
                config to use

        Returns:
            A torch dataloader for `dataset` with parameters configured as specified
        """
        if mode.lower() == "train":
            dataloader_params = self.cfg.dataloader.train_dataloader
        elif mode.lower() == "val":
            dataloader_params = self.cfg.dataloader.val_dataloader
        elif mode.lower() == "test":
            dataloader_params = self.cfg.dataloader.test_dataloader
        else:
            raise ValueError(
                "`mode` must be one of ['train', 'val','test'], not '{mode}'"
            )
        if dataloader_params.num_workers > 0:
            # prevent too many open files error
            pin_memory = True
            torch.multiprocessing.set_sharing_strategy("file_system")
        else:
            pin_memory = False
        
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=1,
            pin_memory=pin_memory,
            collate_fn=dataset.no_batching_fn,
            **dataloader_params
        )

    def get_optimizer(self, params: Iterable) -> torch.optim.Optimizer:
        """Getter for optimizer.

        Args:
            params: iterable of model parameters to optimize or dicts defining
                parameter groups

        Returns:
            A torch Optimizer with specified params
        """
        optimizer_params = self.cfg.optimizer
        return init_optimizer(params, optimizer_params)

    def get_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Getter for lr scheduler.

        Args:
            optimizer: The optimizer to wrap the scheduler around

        Returns:
            A torch learning rate scheduler with specified params
        """
        lr_scheduler_params = self.cfg.scheduler
        return init_scheduler(optimizer, lr_scheduler_params)

    def get_loss(self) -> AssoLoss:
        """Getter for loss functions.

        Returns:
            An AssoLoss with specified params
        """
        loss_params = self.cfg.loss
        return AssoLoss(**loss_params)

    def get_logger(self):
        """Getter for logging callback.

        Returns:
            A Logger with specified params
        """
        logger_params = OmegaConf.to_container(self.cfg.logging, resolve=True)
        return init_logger(logger_params)

    def get_early_stopping(self) -> pl.callbacks.EarlyStopping:
        """Getter for lightning early stopping callback.

        Returns:
            A lightning early stopping callback with specified params
        """
        early_stopping_params = self.cfg.early_stopping
        return pl.callbacks.EarlyStopping(**early_stopping_params)

    def get_checkpointing(self) -> pl.callbacks.ModelCheckpoint:
        """Getter for lightning checkpointing callback.

        Returns:
            A lightning checkpointing callback with specified params
        """
        # convert to dict to enable extracting/removing params
        checkpoint_params = OmegaConf.to_container(self.cfg.checkpointing, resolve=True)
        logging_params = self.cfg.logging
        if "dirpath" not in checkpoint_params or checkpoint_params["dirpath"] is None:
            if "group" in logging_params:
                dirpath = f"./models/{logging_params.group}/{logging_params.name}"
            else:
                dirpath = f"./models/{logging_params.name}"

        else:
            dirpath = checkpoint_params["dirpath"]
        _ = checkpoint_params.pop("dirpath")
        checkpointers = []
        monitor = checkpoint_params.pop("monitor")
        for metric in monitor:
            checkpointer = pl.callbacks.ModelCheckpoint(
                monitor=metric, dirpath=dirpath, **checkpoint_params
            )
            checkpointer.CHECKPOINT_NAME_LAST = f"{{epoch}}-best-{{{metric}}}"
            checkpointers.append(checkpointer)
        return checkpointers

    def get_trainer(
        self,
        callbacks: list[pl.callbacks.Callback],
        logger: pl.loggers.WandbLogger,
        devices: int = 1,
        accelerator: str = None
    ) -> pl.Trainer:
        """Getter for the lightning trainer.

        Args:
            callbacks: a list of lightning callbacks preconfigured to be used
                for training
            logger: the Wandb logger used for logging during training
            devices: The number of gpus to be used. 0 means cpu
            accelerator: either "gpu" or "cpu" specifies which device to use

        Returns:
            A lightning Trainer with specified params
        """
        if "accelerator" not in self.cfg.trainer:
            self.set_hparams({'trainer.accelerator': accelerator})
        if "devices" not in self.cfg.trainer:
            self.set_hparams({'trainer.devices': devices})

        trainer_params = self.cfg.trainer
        
        return pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            **trainer_params,
        )
