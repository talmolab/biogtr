"""Module containing logic for loading datasets."""
from biogtr.datasets import data_utils
from torch.utils.data import Dataset
from typing import List
import torch


class BaseDataset(Dataset):
    """Base Dataset for microscopy and sleap datasets to override."""

    def __init__(
        self,
        files: list[str],
        padding: int,
        crop_size: int,
        chunk: bool,
        clip_length: int,
        mode: str,
        augmentations: dict = None,
        gt_list: str = None,
    ):
        """Initialize Dataset.

        Args:
            files: a list of files, file types are combined in subclasses
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train` or `val`. Determines whether this dataset is used for
                training or validation. Currently doesn't affect dataset logic
            augmentations: An optional dict mapping augmentations to parameters.
                See subclasses for details.
            gt_list: An optional path to .txt file containing ground truth for
                cell tracking challenge datasets.
        """
        self.files = files
        self.padding = padding
        self.crop_size = crop_size
        self.chunk = chunk
        self.clip_length = clip_length
        self.mode = mode

        self.augmentations = (
            data_utils.build_augmentations(augmentations) if augmentations else None
        )

        # Initialize in subclasses
        self.frame_idx = None
        self.labels = None
        self.gt_list = None
        self.chunks = None

    def create_chunks(self):
        """Get indexing for data.

        Creates both indexes for selecting dataset (label_idx) and frame in
        dataset (chunked_frame_idx). If chunking is false, we index directly
        using the frame ids. Setting chunking to true creates a list of lists
        containing chunk frames for indexing. This is useful for computational
        efficiency and data shuffling. To be called by subclass __init__()
        """
        if self.chunk:
            self.chunks = [
                [i * self.clip_length for i in range(len(label) // self.clip_length)]
                for label in self.labels
            ]

            self.chunked_frame_idx, self.label_idx = [], []
            for i, (split, frame_idx) in enumerate(zip(self.chunks, self.frame_idx)):
                frame_idx_split = torch.split(frame_idx, self.clip_length)
                self.chunked_frame_idx.extend(frame_idx_split)
                self.label_idx.extend(len(frame_idx_split) * [i])
        else:
            self.chunked_frame_idx = self.frame_idx
            self.label_idx = [i for i in range(len(self.labels))]

    def __len__(self):
        """Get the size of the dataset.

        Returns:
            the size or the number of chunks in the dataset
        """
        return len(self.chunked_frame_idx)

    def no_batching_fn(self, batch):
        """Collate function used to overwrite dataloader batching function.

        Args:
            batch: the chunk of frames to be returned

        Returns:
            The batch
        """
        return batch

    def __getitem__(self, idx: int) -> List[dict]:
        """Get an element of the dataset.

        Args:
            idx: the index of the batch. Note this is not the index of the video
            or the frame.

        Returns:
            A list of dicts where each dict corresponds a frame in the chunk and
            each value is a `torch.Tensor`. Dict elements can be seen in
            subclasses

        """
        label_idx, frame_idx = self.get_indices(idx)

        return self.get_instances(label_idx, frame_idx)

    def get_indices(self, idx: int):
        """Retrieves label and frame indices given batch index.

        This method should be implemented in any subclass of the BaseDataset.

        Args:
            idx: the index of the batch.

        Raises:
            NotImplementedError: If this method is not overridden in a subclass.
        """
        raise NotImplementedError("Must be implemented in subclass")

    def get_instances(self, label_idx: List[int], frame_idx: List[int]):
        """Builds instances dict given label and frame indices.

        This method should be implemented in any subclass of the BaseDataset.

        Args:
            label_idx: The index of the labels.
            frame_idx: The index of the frames.

        Raises:
            NotImplementedError: If this method is not overridden in a subclass.
        """
        raise NotImplementedError("Must be implemented in subclass")
