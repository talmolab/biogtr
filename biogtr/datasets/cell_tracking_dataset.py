"""Module containing cell tracking challenge dataset."""
from PIL import Image
from biogtr.datasets import data_utils
from biogtr.datasets.base_dataset import BaseDataset
from biogtr.data_structures import Instance, Frame
from scipy.ndimage import measurements
from typing import List, Optional, Union
import albumentations as A
import numpy as np
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
        gt_list: str = None,
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
            gt_list,
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
        """Retrieve label and frame indices given batch index.

        Args:
            idx: the index of the batch.
        """
        return self.label_idx[idx], self.chunked_frame_idx[idx]

    def get_instances(self, label_idx: List[int], frame_idx: List[int]) -> List[Frame]:
        """Get an element of the dataset.

        Args:
            label_idx: index of the labels
            frame_idx: index of the frames

        Returns:
            a list of Frame objects containing frame metadata and Instance Objects.
            See `biogtr.data_structures` for more info.
        """
        image = self.videos[label_idx]
        gt = self.labels[label_idx]

        frames = []

        for i in frame_idx:
            instances, gt_track_ids, centroids, bboxes = [], [], [], []

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

            for i in range(len(gt_track_ids)):
                crop = data_utils.crop_bbox(img, bboxes[i])

                instances.append(
                    Instance(
                        gt_track_id=gt_track_ids[i],
                        pred_track_id=-1,
                        bbox=bboxes[i],
                        crop=crop,
                    )
                )

            frames.append(
                Frame(
                    video_id=label_idx,
                    frame_id=i,
                    img_shape=img.shape,
                    instances=instances,
                )
            )

        return frames
