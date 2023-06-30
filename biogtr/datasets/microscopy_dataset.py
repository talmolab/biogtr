"""Module containing microscopy dataset."""
from PIL import Image
from biogtr.datasets import data_utils
from biogtr.datasets.base_dataset import BaseDataset
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf
from typing import Optional
import albumentations as A
import numpy as np
import random
import torch


class MicroscopyDataset(BaseDataset):
    """Dataset for loading Microscopy Data."""

    def __init__(
        self,
        videos: list[str],
        tracks: list[str],
        source: str,
        padding: int = 5,
        crop_size: int = 20,
        chunk: bool = False,
        clip_length: int = 10,
        mode: str = "Train",
        augmentations: Optional[dict] = None,
    ):
        """Initialize MicroscopyDataset.

        Args:
            videos: paths to raw microscopy videos
            tracks: paths to trackmate gt labels (either .xml or .csv)
            source: file format of gt labels based on label generator
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train` or `val`. Determines whether this dataset is used for
                training or validation. Currently doesn't affect dataset logic
            augmentations: An optional dict mapping augmentations to parameters. The keys
                should map directly to augmentation classes in albumentations. Example:
                    augs = {
                        'Rotate': {'limit': [-90, 90]},
                        'GaussianBlur': {'blur_limit': (3, 7), 'sigma_limit': 0},
                        'RandomContrast': {'limit': 0.2}
                    }
        """
        super().__init__(
            videos + tracks, padding, crop_size, chunk, clip_length, mode, augmentations
        )

        self.videos = videos
        self.tracks = tracks
        self.chunk = chunk
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.padding = padding
        self.mode = mode

        self.augmentations = (
            data_utils.build_augmentations(augmentations) if augmentations else None
        )

        if source.lower() == "trackmate":
            parser = data_utils.parse_trackmate
        elif source.lower() in ["icy", "isbi"]:
            parser = lambda x: data_utils.parse_synthetic(x, source=source)
        else:
            raise ValueError(
                f"{source} is unsupported! Must be one of [trackmate, icy, isbi]"
            )

        self.labels = [
            parser(self.tracks[video_idx]) for video_idx in range(len(self.tracks))
        ]

        self.frame_idx = [
            torch.arange(Image.open(video).n_frames)
            if type(video) == str
            else torch.arange(len(video))
            for video in self.videos
        ]

        # Method in BaseDataset. Creates label_idx and chunked_frame_idx to be
        # used in call to get_instances()
        self.create_chunks()

    def get_indices(self, idx):
        """Retrieves label and frame indices given batch index.

        Args:
            idx: the index of the batch.
        """
        return self.label_idx[idx], self.chunked_frame_idx[idx]

    def get_instances(self, label_idx: list[int], frame_idx: list[int]) -> list[dict]:
        """Get an element of the dataset.

        Args:
            label_idx: index of the labels
            frame_idx: index of the frames

        Returns:
            a list of dicts where each dict corresponds a frame in the chunk
            and each value is a `torch.Tensor`.

            Dict Elements:
                {
                    "video_id": The video being passed through the transformer,
                    "img_shape": the shape of each frame,
                    "frame_id": the specific frame in the entire video being used,
                    "num_detected": The number of objects in the frame,
                    "gt_track_ids": The ground truth labels,
                    "bboxes": The bounding boxes of each object,
                    "crops": The raw pixel crops,
                    "features": The feature vectors for each crop outputed by the
                        CNN encoder,
                    "pred_track_ids": The predicted trajectory labels from the
                        tracker,
                    "asso_output": the association matrix preprocessing,
                    "matches": the true positives from the model,
                    "traj_score": the association matrix post processing,
                }
        """
        labels = self.labels[label_idx]
        labels = labels.dropna(how="all")

        video = self.videos[label_idx]

        if type(video) != list:
            video = data_utils.LazyTiffStack(self.videos[label_idx])

        instances = []

        for i in frame_idx:
            gt_track_ids, centroids, bboxes, crops = [], [], [], []

            img = (
                video.get_section(i)
                if type(video) != list
                else np.array(Image.open(video[i]))
            )

            lf = labels[labels["FRAME"].astype(int) == i.item()]

            for instance in sorted(lf["TRACK_ID"].unique()):
                gt_track_ids.append(int(instance))

                x = lf[lf["TRACK_ID"] == instance]["POSITION_X"].iloc[0]
                y = lf[lf["TRACK_ID"] == instance]["POSITION_Y"].iloc[0]
                centroids.append([x, y])

            # albumentations wants (spatial, channels), ensure correct dims
            if self.augmentations is not None:
                for transform in self.augmentations:
                    # for occlusion simulation, can remove if we don't want
                    if isinstance(transform, A.CoarseDropout):
                        transform.fill_value = random.randint(0, 255)

                augmented = self.augmentations(
                    image=img,
                    keypoints=np.vstack(centroids),
                )
                img, centroids = augmented["image"], augmented["keypoints"]

            img = torch.Tensor(img)

            # torch wants (channels, spatial) - ensure correct dims
            if len(img.shape) == 2:
                img = img.unsqueeze(0)
            elif len(img.shape) == 3:
                if img.shape[2] == 3:
                    img = img.T  # todo: check for edge cases

            for c in centroids:
                bbox = data_utils.pad_bbox(
                    data_utils.get_bbox([int(c[0]), int(c[1])], self.crop_size),
                    padding=self.padding,
                )
                crop = data_utils.crop_bbox(img, bbox)

                bboxes.append(bbox)
                crops.append(crop)

            instances.append(
                {
                    "video_id": torch.tensor([label_idx]),
                    "img_shape": torch.tensor([img.shape]),
                    "frame_id": torch.tensor([i]),
                    "num_detected": torch.tensor([len(bboxes)]),
                    "gt_track_ids": torch.tensor(gt_track_ids).type(torch.int64),
                    "bboxes": torch.stack(bboxes),
                    "crops": torch.stack(crops),
                    "features": torch.tensor([]),
                    "pred_track_ids": torch.tensor([-1 for _ in range(len(bboxes))]),
                    "asso_output": torch.tensor([]),
                    "matches": torch.tensor([]),
                    "traj_score": torch.tensor([]),
                }
            )

        return instances
