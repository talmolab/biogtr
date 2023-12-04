"""Script to run inference and get out tracks."""

from biogtr.config import Config
from biogtr.models.gtr_runner import GTRRunner
from biogtr.data_structures import Frame
from omegaconf import DictConfig
from pprint import pprint
from pathlib import Path

import os
import hydra
import pandas as pd
import pytorch_lightning as pl
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_default_device(device)


def export_trajectories(frames_pred: list[Frame], save_path: str = None):
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
    for frame in frames_pred:
        for i, instance in enumerate(frame.instances):
            frame_ids.append(frame.frame_id.item())
            bbox = instance.bbox.squeeze()
            y = (bbox[2] + bbox[0]) / 2
            x = (bbox[3] + bbox[1]) / 2
            X.append(x.item())
            Y.append(y.item())
            pred_track_ids.append(instance.pred_track_id.item())

    save_dict["Frame"] = frame_ids
    save_dict["X"] = X
    save_dict["Y"] = Y
    save_dict["Pred_track_id"] = pred_track_ids
    save_df = pd.DataFrame(save_dict)
    if save_path:
        save_df.to_csv(save_path, index=False)
    return save_df


def inference(
    model: GTRRunner, dataloader: torch.utils.data.DataLoader
) -> list[pd.DataFrame]:
    """Run Inference.

    Args:
        model: model loaded from checkpoint used for inference
        dataloader: dataloader containing inference data

    Return:
        List of DataFrames containing prediction results for each video
    """
    num_videos = len(dataloader.dataset.slp_files)
    trainer = pl.Trainer(devices=1, limit_predict_batches=3)
    preds = trainer.predict(model, dataloader)

    vid_trajectories = [[] for i in range(num_videos)]

    for batch in preds:
        for frame in batch:
            vid_trajectories[frame.video_id].append(frame)

    saved = []

    for video in vid_trajectories:
        if len(video) > 0:
            save_dict = {}
            video_ids = []
            frame_ids = []
            X, Y = [], []
            pred_track_ids = []
            for frame in video:
                for i, instance in frame.instances:
                    video_ids.append(frame.video_id.item())
                    frame_ids.append(frame.frame_id.item())
                    bbox = instance.bbox
                    y = (bbox[2] + bbox[0]) / 2
                    x = (bbox[3] + bbox[1]) / 2
                    X.append(x.item())
                    Y.append(y.item())
                    pred_track_ids.append(instance.pred_track_id.item())
            save_dict["Video"] = video_ids
            save_dict["Frame"] = frame_ids
            save_dict["X"] = X
            save_dict["Y"] = Y
            save_dict["Pred_track_id"] = pred_track_ids
            save_df = pd.DataFrame(save_dict)
            saved.append(save_df)

    return saved


@hydra.main(config_path="configs", config_name=None, version_base=None)
def main(cfg: DictConfig):
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
            print("Pod Index Not found! Setting index to 0")
            index = 0
        print(f"Pod Index: {index}")

        checkpoints = pd.read_csv(cfg.checkpoints)
        checkpoint = checkpoints.iloc[index]
    else:
        checkpoint = pred_cfg.get_ckpt_path()

    model = GTRRunner.load_from_checkpoint(checkpoint)
    tracker_cfg = pred_cfg.get_tracker_cfg()
    print("Updating tracker hparams")
    model.tracker_cfg = tracker_cfg
    print(f"Using the following params for tracker:")
    pprint(model.tracker_cfg)
    dataset = pred_cfg.get_dataset(mode="test")

    dataloader = pred_cfg.get_dataloader(dataset, mode="test")
    preds = inference(model, dataloader)
    for i, pred in enumerate(preds):
        print(pred)
        outdir = pred_cfg.cfg.outdir if "outdir" in pred_cfg.cfg else "./results"
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(
            outdir,
            f"{Path(pred_cfg.cfg.dataset.test_dataset.slp_files[i]).stem}_tracking_results",
        )
        print(f"Saving to {outpath}")
        # TODO: Figure out how to overwrite sleap labels instance labels w pred instance labels then save as a new slp file
        pred.to_csv(outpath, index=False)


if __name__ == "__main__":
    main()
