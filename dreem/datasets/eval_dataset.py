"""Module containing wrapper for merging gt and pred datasets for evaluation."""

from torch.utils.data import Dataset
from dreem.io import Instance, Frame


class EvalDataset(Dataset):
    """Wrapper around gt and predicted dataset."""

    def __init__(self, gt_dataset: Dataset, pred_dataset: Dataset) -> None:
        """Initialize EvalDataset.

        Args:
            gt_dataset: A Dataset object containing ground truth track ids
            pred_dataset: A dataset object containing predicted track ids
        """
        self.gt_dataset = gt_dataset
        self.pred_dataset = pred_dataset

    def __len__(self) -> int:
        """Get the size of the dataset.

        Returns:
            the size or the number of chunks in the dataset
        """
        return len(self.gt_dataset)

    def __getitem__(self, idx: int) -> list[Frame]:
        """Get an element of the dataset.

        Args:
            idx: the index of the batch. Note this is not the index of the video
                or the frame.

        Returns:
            A list of Frames where frames contain instances w gt and pred track ids + bboxes.
        """
        gt_batch = self.gt_dataset[idx]
        pred_batch = self.pred_dataset[idx]

        eval_frames = []
        for gt_frame, pred_frame in zip(gt_batch, pred_batch):
            eval_instances = []
            for i, gt_instance in enumerate(gt_frame.instances):

                gt_track_id = gt_instance.gt_track_id

                try:
                    pred_track_id = pred_frame.instances[i].gt_track_id
                    pred_bbox = pred_frame.instances[i].bbox
                except IndexError:
                    pred_track_id = -1
                    pred_bbox = [-1, -1, -1, -1]
                eval_instances.append(
                    Instance(
                        gt_track_id=gt_track_id,
                        pred_track_id=pred_track_id,
                        bbox=pred_bbox,
                    )
                )
            eval_frames.append(
                Frame(
                    video_id=gt_frame.video_id,
                    frame_id=gt_frame.frame_id,
                    vid_file=gt_frame.video.filename,
                    img_shape=gt_frame.img_shape,
                    instances=eval_instances,
                )
            )

        return eval_frames
