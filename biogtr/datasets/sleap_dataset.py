"""Module containing logic for loading sleap datasets."""
import albumentations as A
import torch
import imageio
import numpy as np
import sleap_io as sio
import random
from biogtr.datasets import data_utils
from biogtr.datasets.base_dataset import BaseDataset
from torchvision.transforms import functional as tvf
from typing import List, Union


class SleapDataset(BaseDataset):
    """Dataset for loading animal behavior data from sleap."""

    def __init__(
        self,
        slp_files: list[str],
        video_files: list[str],
        padding: int = 5,
        crop_size: int = 128,
        chunk: bool = True,
        clip_length: int = 500,
        mode: str = "train",
        augmentations: dict = None,
        n_chunks: Union[int, float] = 1.0,
        seed: int = None,
    ):
        """Initialize SleapDataset.

        Args:
            slp_files: a list of .slp files storing tracking annotations
            video_files: a list of paths to video files
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train` or `val`. Determines whether this dataset is used for
                training or validation. Currently doesn't affect dataset logic
            augmentations: An optional dict mapping augmentations to parameters. The keys
                should map directly to augmentation classes in albumentations. Example:
                    augmentations = {
                        'Rotate': {'limit': [-90, 90], 'p': 0.5},
                        'GaussianBlur': {'blur_limit': (3, 7), 'sigma_limit': 0, 'p': 0.2},
                        'RandomContrast': {'limit': 0.2, 'p': 0.6}
                    }
            n_chunks: Number of chunks to subsample from.
                Can either a fraction of the dataset (ie (0,1.0]) or number of chunks
            seed: set a seed for reproducibility
        """
        super().__init__(
            slp_files + video_files,
            padding,
            crop_size,
            chunk,
            clip_length,
            mode,
            augmentations,
            n_chunks,
            seed,
        )

        self.slp_files = slp_files
        self.video_files = video_files
        self.padding = padding
        self.crop_size = crop_size
        self.chunk = chunk
        self.clip_length = clip_length
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

        self.labels = [sio.load_slp(slp_file) for slp_file in self.slp_files]

        # do we need this? would need to update with sleap-io

        # for label in self.labels:
        # label.remove_empty_instances(keep_empty_frames=False)

        self.anchor_names = [
            data_utils.sorted_anchors(labels) for labels in self.labels
        ]

        self.frame_idx = [torch.arange(len(label)) for label in self.labels]

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
            A list of dicts where each dict corresponds a frame in the chunk and each value is a `torch.Tensor`
            Dict Elements:
            {
                        "video_id": The video being passed through the transformer,
                        "img_shape": the shape of each frame,
                        "frame_id": the specific frame in the entire video being used,
                        "num_detected": The number of objects in the frame,
                        "gt_track_ids": The ground truth labels,
                        "bboxes": The bounding boxes of each object,
                        "crops": The raw pixel crops,
                        "features": The feature vectors for each crop outputed by the CNN encoder,
                        "pred_track_ids": The predicted trajectory labels from the tracker,
                        "asso_output": the association matrix preprocessing,
                        "matches": the true positives from the model,
                        "traj_score": the association matrix post processing,
                }

        """
        video = self.labels[label_idx]

        anchors = [
            video.skeletons[0].node_names.index(anchor_name)
            for anchor_name in self.anchor_names[label_idx]
        ]

        video_name = self.video_files[label_idx]

        vid_reader = imageio.get_reader(video_name, "ffmpeg")

        img = vid_reader.get_data(0)
        crop_shape = (img.shape[-1], *(self.crop_size + 2 * self.padding,) * 2)

        instances = []

        for i in frame_idx:
            gt_track_ids, bboxes, crops, poses, shown_poses = [], [], [], [], []

            i = int(i)

            lf = video[i]
            img = vid_reader.get_data(i)

            for instance in lf:
                gt_track_ids.append(video.tracks.index(instance.track))

                poses.append(
                    dict(
                        zip(
                            [n.name for n in instance.skeleton.nodes],
                            np.array(instance.numpy()).tolist(),
                        )
                    )
                )

                shown_poses.append(
                    dict(
                        zip(
                            [n.name for n in instance.skeleton.nodes],
                            [[p.x, p.y] for p in instance.points.values()],
                        )
                    )
                )

            # augmentations
            if self.augmentations is not None:
                for transform in self.augmentations:
                    if isinstance(transform, A.CoarseDropout):
                        transform.fill_value = random.randint(0, 255)

                if shown_poses:
                    keypoints = np.vstack([list(s.values()) for s in shown_poses])

                else:
                    keypoints = []

                augmented = self.augmentations(image=img, keypoints=keypoints)

                img, aug_poses = augmented["image"], augmented["keypoints"]

                aug_poses = [
                    arr
                    for arr in np.split(
                        np.array(aug_poses),
                        np.array([len(s) for s in shown_poses]).cumsum(),
                    )
                    if arr.size != 0
                ]

                aug_poses = [
                    dict(zip(list(pose_dict.keys()), aug_pose_arr.tolist()))
                    for aug_pose_arr, pose_dict in zip(aug_poses, shown_poses)
                ]

                _ = [pose.update(aug_pose) for pose, aug_pose in zip(poses, aug_poses)]

            img = tvf.to_tensor(img)

            for pose in poses:
                bbox = data_utils.pad_bbox(
                    data_utils.centroid_bbox(
                        np.array(list(pose.values())), anchors, self.crop_size
                    ),
                    padding=self.padding,
                )

                crop = data_utils.crop_bbox(img, bbox)

                bboxes.append(bbox)
                crops.append(crop)

            stacked_crops = (
                torch.stack(crops) if crops else torch.empty((0, *crop_shape))
            )

            instances.append(
                {
                    "video_id": torch.tensor([label_idx]),
                    "img_shape": torch.tensor([img.shape]),
                    "frame_id": torch.tensor([i]),
                    "num_detected": torch.tensor([len(bboxes)]),
                    "gt_track_ids": torch.tensor(gt_track_ids),
                    "bboxes": torch.stack(bboxes) if bboxes else torch.empty((0, 4)),
                    "crops": stacked_crops,
                    "features": torch.tensor([]),
                    "pred_track_ids": torch.tensor([-1 for _ in range(len(bboxes))]),
                    "asso_output": torch.tensor([]),
                    "matches": torch.tensor([]),
                    "traj_score": torch.tensor([]),
                }
            )

        return instances
