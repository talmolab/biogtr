"""Script to run inference and get out tracks."""

from dreem.io import Config
from dreem.models import GTRRunner
from omegaconf import DictConfig
from pathlib import Path

import hydra
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import sleap_io as sio
import logging

logger = logging.getLogger("dreem.inference")


def export_trajectories(
    frames_pred: list["dreem.io.Frame"], save_path: str | None = None
) -> pd.DataFrame:
    """Convert trajectories to data frame and save as .csv.

    Args:
        frames_pred: A list of Frames with predicted track ids.
        save_path: The path to save the predicted trajectories to.

    Returns:
        A dictionary containing the predicted track id and centroid coordinates for each instance in the video.
    """
    save_dict = {}
    frame_ids = []
    X, Y = [], []
    pred_track_ids = []
    track_scores = []
    for frame in frames_pred:
        for i, instance in enumerate(frame.instances):
            frame_ids.append(frame.frame_id.item())
            bbox = instance.bbox.squeeze()
            y = (bbox[2] + bbox[0]) / 2
            x = (bbox[3] + bbox[1]) / 2
            X.append(x.item())
            Y.append(y.item())
            track_scores.append(instance.track_score)
            pred_track_ids.append(instance.pred_track_id.item())

    save_dict["Frame"] = frame_ids
    save_dict["X"] = X
    save_dict["Y"] = Y
    save_dict["Pred_track_id"] = pred_track_ids
    save_dict["Track_score"] = track_scores
    save_df = pd.DataFrame(save_dict)
    if save_path:
        save_df.to_csv(save_path, index=False)
    return save_df


def track(
    model: GTRRunner, trainer: pl.Trainer, dataloader: torch.utils.data.DataLoader
) -> list[pd.DataFrame]:
    """Run Inference.

    Args:
        model: GTRRunner model loaded from checkpoint used for inference
        trainer: lighting Trainer object used for handling inference log.
        dataloader: dataloader containing inference data

    Return:
        List of DataFrames containing prediction results for each video
    """
    num_videos = len(dataloader.dataset.vid_files)
    preds = trainer.predict(model, dataloader)

    vid_trajectories = {i: [] for i in range(num_videos)}

    tracks = {}
    for batch in preds:
        for frame in batch:
            lf, tracks = frame.to_slp(tracks)
            if frame.frame_id.item() == 0:
                logger.info(f"Video: {lf.video}")
            vid_trajectories[frame.video_id.item()].append(lf)

    for vid_id, video in vid_trajectories.items():
        if len(video) > 0:

            vid_trajectories[vid_id] = sio.Labels(video)

    return vid_trajectories


@hydra.main(config_path="configs", config_name=None, version_base=None)
def run(cfg: DictConfig) -> dict[int, sio.Labels]:
    """Run inference based on config file.

    Args:
        cfg: A dictconfig loaded from hydra containing checkpoint path and data
    """
    pred_cfg = Config(cfg)

    if "checkpoints" in cfg.keys():
        try:
            index = int(os.environ["POD_INDEX"])
        # For testing without deploying a job on runai
        except KeyError:
            index = input("Pod Index Not found! Please choose a pod index: ")

        logger.info(f"Pod Index: {index}")

        checkpoints = pd.read_csv(cfg.checkpoints)
        checkpoint = checkpoints.iloc[index]
    else:
        checkpoint = pred_cfg.cfg.ckpt_path

    model = GTRRunner.load_from_checkpoint(checkpoint)
    tracker_cfg = pred_cfg.get_tracker_cfg()
    logger.info("Updating tracker hparams")
    model.tracker_cfg = tracker_cfg
    logger.info(f"Using the following params for tracker:")
    logger.info(model.tracker_cfg)

    dataset = pred_cfg.get_dataset(mode="test")
    dataloader = pred_cfg.get_dataloader(dataset, mode="test")

    trainer = pred_cfg.get_trainer()

    preds = track(model, trainer, dataloader)

    outdir = pred_cfg.cfg.outdir if "outdir" in pred_cfg.cfg else "./results"
    os.makedirs(outdir, exist_ok=True)

    run_num = 0
    for i, pred in preds.items():
        outpath = os.path.join(
            outdir,
            f"{Path(dataloader.dataset.label_files[i]).stem}.dreem_inference.v{run_num}.slp",
        )
        if os.path.exists(outpath):
            run_num += 1
            outpath = outpath.replace(f".v{run_num-1}", f".v{run_num}")
        logger.info(f"Saving {preds} to {outpath}")
        pred.save(outpath)

    return preds


if __name__ == "__main__":
    # example calls:

    # train with base config:
    # python train.py --config-dir=./configs --config-name=inference

    # override with params config:
    # python train.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml

    # override with params config, and specific params:
    # python train.py --config-dir=./configs --config-name=inference +params_config=configs/params.yaml dataset.train_dataset.padding=10
    run()
