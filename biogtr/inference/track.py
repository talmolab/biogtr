"""Script to run inference and get out tracks."""

from biogtr.config import Config
from biogtr.datasets.tracking_dataset import TrackingDataset
from biogtr.models.gtr_runner import GTRRunner
from omegaconf import DictConfig
from pathlib import Path
from pprint import pprint
import hydra
import os
import pandas as pd
import pytorch_lightning as pl
import torch


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
            vid_trajectories[frame["video_id"]].append(frame)

    saved = []

    for video in vid_trajectories:
        if len(video) > 0:
            save_dict = {}
            video_ids = []
            frame_ids = []
            X, Y = [], []
            pred_track_ids = []
            for frame in video:
                for i in range(frame["num_detected"]):
                    video_ids.append(frame["video_id"].item())
                    frame_ids.append(frame["frame_id"].item())
                    bbox = frame["bboxes"][i]

                    y = (bbox[2] + bbox[0]) / 2
                    x = (bbox[3] + bbox[1]) / 2
                    X.append(x.item())
                    Y.append(y.item())
                    pred_track_ids.append(frame["pred_track_ids"][i].item())
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
    """Main function for running inference.

    handles config parsing, batch deployment and saving results

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
