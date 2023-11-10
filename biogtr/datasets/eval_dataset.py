"Module containing wrapper for merging gt and pred datasets for evaluation"
import torch
from torch.utils.data import Dataset

class EvalDataset(Dataset):
    
    def __init__(self, gt_dataset: Dataset, pred_dataset: Dataset):
        
        self.gt_dataset = gt_dataset
        self.pred_dataset = pred_dataset
        
    def __len__(self):
        """Get the size of the dataset.

        Returns:
            the size or the number of chunks in the dataset
        """
        return len(self.gt_dataset)
        
    def __getitem__(self, idx: int):
        """Get an element of the dataset.

        Args:
            idx: the index of the batch. Note this is not the index of the video
            or the frame.

        Returns:
            A list of dicts where each dict corresponds a frame in the chunk and
            each value is a `torch.Tensor`. Dict elements are the video id, frame id, and gt/pred track ids

        """
        labels = [{"video_id": gt_frame['video_id'],
                  "frame_id": gt_frame['video_id'],
                  "gt_track_ids": gt_frame['gt_track_ids'],
                  "pred_track_ids": pred_frame['gt_track_ids'],
                  "bboxes": pred_frame["bboxes"]
                 } for gt_frame, pred_frame in zip(self.gt_dataset[idx], self.pred_dataset[idx])]
        
        return labels