"""Module containing microscopy dataset."""

from PIL import Image
from dreem.datasets import data_utils, BaseDataset
from dreem.io import Instance, Frame
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
        augmentations: dict | None = None,
        n_chunks: int | float = 1.0,
        seed: int | None = None,
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
            n_chunks: Number of chunks to subsample from.
                Can either a fraction of the dataset (ie (0,1.0]) or number of chunks
            seed: set a seed for reproducibility
        """
        super().__init__(
            tracks,
            videos,
            padding,
            crop_size,
            chunk,
            clip_length,
            mode,
            augmentations,
            n_chunks,
            seed,
        )

        self.vid_files = videos
        self.tracks = tracks
        self.chunk = chunk
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.padding = padding
        self.mode = mode.lower()
        self.n_chunks = n_chunks
        self.seed = seed

        # if self.seed is not None:
        #     np.random.seed(self.seed)
        if augmentations and self.mode == "train":
            self.augmentations = data_utils.build_augmentations(augmentations)
        else:
            self.augmentations = None

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

        self.videos = []
        for vid_file in self.vid_files:
            if not isinstance(vid_file, list):
                self.videos.append(data_utils.LazyTiffStack(vid_file))
            else:
                self.videos.append([Image.open(frame_file) for frame_file in vid_file])
        self.frame_idx = [
            (
                torch.arange(Image.open(video).n_frames)
                if isinstance(video, str)
                else torch.arange(len(video))
            )
            for video in self.vid_files
        ]

        # Method in BaseDataset. Creates label_idx and chunked_frame_idx to be
        # used in call to get_instances()
        self.create_chunks()

    def get_indices(self, idx: int) -> tuple:
        """Retrieve label and frame indices given batch index.

        Args:
            idx: the index of the batch.
        """
        return self.label_idx[idx], self.chunked_frame_idx[idx]

    def get_instances(self, label_idx: list[int], frame_idx: list[int]) -> list[Frame]:
        """Get an element of the dataset.

        Args:
            label_idx: index of the labels
            frame_idx: index of the frames

        Returns:
            A list of Frames containing Instances to be tracked (See `dreem.io.data_structures for more info`)
        """
        labels = self.labels[label_idx]
        labels = labels.dropna(how="all")

        video = self.videos[label_idx]

        frames = []
        for frame_id in frame_idx:
            instances, gt_track_ids, centroids = [], [], []

            img = (
                video.get_section(frame_id)
                if not isinstance(video, list)
                else np.array(video[frame_id])
            )

            lf = labels[labels["FRAME"].astype(int) == frame_id.item()]

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

            for gt_id in range(len(gt_track_ids)):
                c = centroids[gt_id]
                bbox = data_utils.pad_bbox(
                    data_utils.get_bbox([int(c[0]), int(c[1])], self.crop_size),
                    padding=self.padding,
                )
                crop = data_utils.crop_bbox(img, bbox)

                instances.append(
                    Instance(
                        gt_track_id=gt_track_ids[gt_id],
                        pred_track_id=-1,
                        bbox=bbox,
                        crop=crop,
                    )
                )

            if self.mode == "train":
                np.random.shuffle(instances)

            frames.append(
                Frame(
                    video_id=label_idx,
                    frame_id=frame_id,
                    img_shape=img.shape,
                    instances=instances,
                )
            )

        return frames

    def __del__(self):
        """Handle file closing before deletion."""
        for vid_reader in self.videos:
            if not isinstance(vid_reader, list):
                vid_reader.close()
            else:
                for frame_reader in vid_reader:
                    frame_reader.close()
