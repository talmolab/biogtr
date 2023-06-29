"""Training script for training model.

Used for training a single model or deploying a batch train job on RUNAI CLI
"""
from biogtr.config import Config
from biogtr.datasets.tracking_dataset import TrackingDataset
from biogtr.datasets.data_utils import view_training_batch
from biogtr.models.model_utils import get_device
from multiprocessing import cpu_count
from omegaconf import DictConfig
from pprint import pprint
import os
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.multiprocessing

# useful for longer training runs, but not for single iteration debugging
# finds optimal hardware algs which has upfront time increase for first
# iteration, quicker subsequent iterations

# torch.backends.cudnn.benchmark = True

device = get_device()


@hydra.main(config_path="configs", config_name=None, version_base=None)
def main(cfg: DictConfig):
    """Main function for training.

    Handles all config parsing and initialization then calls `trainer.train()`.
    If `batch_config` is included then run will be assumed to be a batch job.

    Args:
        cfg: The config dict parsed by `hydra`
    """
    train_cfg = Config(cfg)

    # update with parameters for batch train job
    if "batch_config" in cfg.keys():
        try:
            index = int(os.environ["POD_INDEX"])
        # For testing without deploying a job on runai
        except KeyError:
            print("Pod Index Not found! Setting index to 0")
            index = 0
        print(f"Pod Index: {index}")

        hparams_df = pd.read_csv(cfg.batch_config)
        hparams = hparams_df.iloc[index].to_dict()

        if train_cfg.set_hparams(hparams):
            print("Updated the following hparams to the following values")
            pprint(hparams)
    else:
        hparams = {}
    pprint(f"Final train config: {train_cfg}")

    model = train_cfg.get_model()
    train_dataset = train_cfg.get_dataset(mode="train")
    train_dataloader = train_cfg.get_dataloader(train_dataset, mode="train")

    val_dataset = train_cfg.get_dataset(mode="val")
    val_dataloader = train_cfg.get_dataloader(val_dataset, mode="val")

    test_dataset = train_cfg.get_dataset(mode="test")
    test_dataloader = train_cfg.get_dataloader(test_dataset, mode="test")

    dataset = TrackingDataset(
        train_dl=train_dataloader, val_dl=val_dataloader, test_dl=test_dataloader
    )

    if cfg.view_batch.enable:
        instances = next(iter(train_dataset))
        view_training_batch(instances, num_frames=cfg.view_batch.num_frames)

        if cfg.view_batch.no_train:
            return

    model = train_cfg.get_gtr_runner()

    logger = train_cfg.get_logger()

    callbacks = []
    _ = callbacks.extend(train_cfg.get_checkpointing())
    _ = callbacks.append(pl.callbacks.LearningRateMonitor())
    _ = callbacks.append(train_cfg.get_early_stopping())

    if device == "cuda":
        accelerator = "gpu"
        devices = torch.cuda.device_count()
    elif device == "mps":
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = cpu_count()

    trainer = train_cfg.get_trainer(
        callbacks,
        logger,
        accelerator=accelerator,
        devices=devices,
    )

    ckpt_path = train_cfg.get_ckpt_path()
    trainer.fit(model, dataset, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py --config-dir=./configs --config-name=base

    # override with params config:
    # python train.py --config-dir=./configs --config-name=base +params_config=configs/params.yaml

    # override with params config, and specific params:
    # python train.py --config-dir=./configs --config-name=base +params_config=configs/params.yaml model.norm=True model.decoder_self_attn=True dataset.padding=10

    # deploy batch train job:
    # python train.py --config-dir=./configs --config-name=base +batch_config=test_batch_train.csv

    main()
