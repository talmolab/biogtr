"""Module containing cell tracking challenge dataset."""
from PIL import Image
from biogtr.datasets import data_utils
from biogtr.datasets.base_dataset import BaseDataset
from scipy.ndimage import measurements
from torch.utils.data import Dataset
from torchvision.transforms import functional as tvf
from typing import List, Optional, Union
import albumentations as A
import glob
import numpy as np
import os
import pandas as pd
import random
import torch


class CellTrackingDataset(BaseDataset):
    """Dataset for loading cell tracking challenge data."""

    def __init__(
        self,
        raw_images: list[str],
        gt_images: list[str],
        padding: int = 5,
        crop_size: int = 20,
        chunk: bool = False,
        clip_length: int = 10,
        mode: str = "train",
        augmentations: Optional[dict] = None,
        n_chunks: Union[int, float] = 1.0,
        seed: int = None,
        gt_list: str = None
    ):
        """Initialize CellTrackingDataset.

        Args:
            raw_images: paths to raw microscopy images
            gt_images: paths to gt label images
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
            n_chunks: Number of chunks to subsample from.
                Can either a fraction of the dataset (ie (0,1.0]) or number of chunks
            seed: set a seed for reproducibility
            gt_list: An optional path to .txt file containing gt ids stored in cell
                tracking challenge format: "track_id", "start_frame",
                "end_frame", "parent_id"
        """
        super().__init__(
            raw_images + gt_images,
            padding,
            crop_size,
            chunk,
            clip_length,
            mode,
            augmentations,
            n_chunks,
            seed,
            gt_list
        )

        self.videos = raw_images
        self.labels = gt_images
        self.chunk = chunk
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.padding = padding
        self.mode = mode
        self.n_chunks = n_chunks
        self.seed = seed

        if self.n_chunks > 1.0:
            self.n_chunks = int(self.n_chunks)

        # if self.seed is not None:
        #     np.random.seed(self.seed)

        self.augmentations = (
            data_utils.build_augmentations(augmentations) if augmentations else None
        )

        if gt_list is not None:
            self.gt_list = pd.read_csv(
                gt_list,
                delimiter=" ",
                header=None,
                names=["track_id", "start_frame", "end_frame", "parent_id"],
            )
        else:
            self.gt_list = None

        self.frame_idx = [torch.arange(len(image)) for image in self.labels]

        # Method in BaseDataset. Creates label_idx and chunked_frame_idx to be
        # used in call to get_instances()
        self.create_chunks()

    def get_indices(self, idx):
        """Retrieves label and frame indices given batch index.

        Args:
            idx: the index of the batch.
        """
        return self.label_idx[idx], self.chunked_frame_idx[idx]

    def get_instances(self, label_idx: List[int], frame_idx: List[int]) -> list[dict]:
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
        image = self.videos[label_idx]
        gt = self.labels[label_idx]

        instances = []

        for i in frame_idx:
            gt_track_ids, centroids, bboxes, crops = [], [], [], []

            i = int(i)

            img = image[i]
            gt_sec = gt[i]

            img = np.array(Image.open(img))
            gt_sec = np.array(Image.open(gt_sec))

            if img.dtype == np.uint16:
                img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype(
                    np.uint8
                )

            if self.gt_list is None:
                unique_instances = np.unique(gt_sec)
            else:
                unique_instances = self.gt_list["track_id"].unique()

            for instance in unique_instances:
                # not all instances are in the frame, and they also label the
                # background instance as zero
                if instance in gt_sec and instance != 0:
                    mask = gt_sec == instance
                    center_of_mass = measurements.center_of_mass(mask)

                    # scipy returns yx
                    x, y = center_of_mass[::-1]

                    bbox = data_utils.pad_bbox(
                        data_utils.get_bbox([int(x), int(y)], self.crop_size),
                        padding=self.padding,
                    )

                    gt_track_ids.append(int(instance))
                    centroids.append([x, y])
                    bboxes.append(bbox)

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

            img = torch.Tensor(img).unsqueeze(0)

            for bbox in bboxes:
                crop = data_utils.crop_bbox(img, bbox)
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
