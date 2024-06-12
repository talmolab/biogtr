"""Module containing different loss functions to be optimized."""

from dreem.models.model_utils import get_boxes, get_times
from torch import nn
import torch
import torch.nn.functional as F
import torchvision

# todo: use named tensors
# todo: clean up names, comments


class AssoLoss(nn.Module):
    """Default association loss used for training GTR model."""

    def __init__(
        self,
        neg_unmatched: bool = False,
        epsilon: float = 1e-4,
        asso_weight: float = 1.0,
    ):
        """Initialize Loss function.

        Args:
            neg_unmatched: Whether or not to set unmatched objects to background
            epsilon: small number used for numeric precision to prevent dividing by zero
            asso_weight: How much to weight the association loss by
        """
        super().__init__()

        self.neg_unmatched = neg_unmatched
        self.epsilon = epsilon
        self.asso_weight = asso_weight

    def forward(
        self, asso_preds: list[torch.Tensor], frames: list["Frame"]
    ) -> torch.Tensor:
        """Calculate association loss.

        Args:
            asso_preds: a list containing the association matrix at each frame
            frames: a list of Frames containing gt labels.

        Returns:
            the association loss between predicted association and actual
        """
        # get number of detected objects and ground truth ids
        n_t = [frame.num_detected for frame in frames]
        target_inst_id = torch.cat(
            [frame.get_gt_track_ids().to(asso_preds[-1].device) for frame in frames]
        )
        instances = [instance for frame in frames for instance in frame.instances]

        # for now set equal since detections are fixed
        pred_box = get_boxes(instances)
        pred_time, _ = get_times(instances)
        pred_box = torch.nanmean(pred_box, axis=1)
        target_box, target_time = pred_box, pred_time

        # todo: we should maybe reconsider how we label gt instances. The second
        # criterion will return true on a single instance video, for example.
        # For now we can ignore this since we train on dense labels.

        """
            # Return a 0 loss if any of the 2 criteria are met
            # 1. the video doesn’t have gt bboxes
            # 2. the maximum id is zero

        sum_instance_lengths = sum(len(x) for x in instances)
        max_instance_lengths = max(
            x["gt_track_ids"].max().item() for x in instances if len(x) > 0
        )

        if sum_instance_lengths == 0 or max_instance_lengths == 0:
            print("No bounding boxes detected, returning zero loss")
            print(f"Sum instance lengths: {sum_instance_lengths}")
            print(f"Max instance lengths: {max_instance_lengths}")
            loss = asso_preds[0].new_zeros((1,), dtype=torch.float32)[0]
            return loss
        """

        asso_gt, match_cues = self._get_asso_gt(
            pred_box, pred_time, target_box, target_time, target_inst_id, n_t
        )

        loss = sum(
            [
                self.detr_asso_loss(asso_pred, asso_gt, match_cues, n_t)
                for asso_pred in asso_preds
            ]
        )

        loss *= self.asso_weight

        return loss

    def _get_asso_gt(
        self,
        pred_box: torch.Tensor,
        pred_time: torch.Tensor,
        target_box: torch.Tensor,
        target_time: torch.Tensor,
        target_inst_id: torch.Tensor,
        n_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the association ground truth for a batch.

        Args:
            pred_box: predicted bounding boxes (N x 4)
            pred_time: predicted time intervals (N,)
            target_box: target bounding boxes (N x 4)
            target_time: target time intervals (N,)
            target_inst_id: target instance IDs (N,)
            n_t: number of ground truth instances (N,)

        Returns:
            A tuple containing:
                asso_gt: Ground truth association matrix (K x N) denoting ground
                    truth instances over time
                match_cues: Tensor indicating which instance is assigned to each gt
                    detection (K x 3) or (N,)
        """
        # compute ious over bboxes, ignore pairs with different time stamps
        ious = torchvision.ops.box_iou(pred_box, target_box)
        ious[pred_time[:, None] != target_time[None, :]] = -1.0

        # get unique instance ids
        inst_ids = torch.unique(target_inst_id[target_inst_id > -1])

        # initialize tensors
        K, N = len(inst_ids), len(pred_box)
        match_cues = pred_box.new_full((N,), -1, dtype=torch.long)
        T = len(n_t)
        asso_gt = pred_box.new_zeros((K, T), dtype=torch.long)

        # split ious by frames
        ious_per_frame = ious.split(n_t, dim=0)

        for k, inst_id in enumerate(inst_ids):
            # get ground truth indices, init index
            target_inds = target_inst_id == inst_id
            base_ind = 0

            for t in range(T):
                # get relevant ious
                iou_t = ious_per_frame[t][:, target_inds]

                # if there are no detections, asso_gt = # gt instances at time step
                if iou_t.numel() == 0:
                    asso_gt[k, t] = n_t[t]
                else:
                    # get max iou and index, select positive ious
                    val, inds = iou_t.max(dim=0)
                    ind = inds[val > 0.0]

                    # make sure there is at most one detection
                    assert len(ind) <= 1, f"{target_inst_id} {n_t}"

                    # if there is one detection with pos IOU, select it
                    if len(ind) == 1:
                        obj_ind = ind[0].item()
                        asso_gt[k, t] = obj_ind
                        match_cues[base_ind + obj_ind] = k

                    # otherwise asso_gt = # gt instances at time step
                    else:
                        asso_gt[k, t] = n_t[t]

                base_ind += n_t[t]

        return asso_gt, match_cues

    def detr_asso_loss(
        self,
        asso_pred: torch.Tensor,
        asso_gt: torch.Tensor,
        match_cues: torch.Tensor,
        n_t: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate association loss between predicted and gt boxes.

        Args:
            asso_pred: Association matrix output from the transformer forward
                pass denoting predicted instances over time (M x N)
            asso_gt: Ground truth association matrix (K x N) denoting ground
                truth instances over time
            match_cues: Tensor indicating which instance is assigned to each gt
                detection (K x 3) or (N,)
            n_t: number of ground truth instances (N,)

        Returns:
            loss: association loss normalized by number of objects
        """
        # get matches between preds and gt
        src_inds, target_inds = self._match(asso_pred, asso_gt, match_cues, n_t)

        loss = 0
        num_objs = 0

        zero = asso_pred.new_zeros((asso_pred.shape[0], 1))  # M x 1
        asso_pred_image = asso_pred.split(n_t, dim=1)  # T x [M x n_t]

        for t in range(len(n_t)):
            # add background class
            asso_pred_with_bg = torch.cat(
                [asso_pred_image[t], zero], dim=1
            )  # M x (n_t + 1)

            if self.neg_unmatched:
                # set unmatched preds to background
                asso_gt_t = asso_gt.new_full((asso_pred.shape[0],), float(n_t[t]))  # M
                asso_gt_t[src_inds] = asso_gt[target_inds, t]  # M
            else:
                # keep only unmatched preds
                asso_pred_with_bg = asso_pred_with_bg[src_inds]  # K x (n_t + 1)
                asso_gt_t = asso_gt[target_inds, t]  # K

            num_objs += (asso_gt_t != n_t[t]).float().sum()

            loss += F.cross_entropy(asso_pred_with_bg, asso_gt_t, reduction="none")

        return loss.sum() / (num_objs + self.epsilon)

    @torch.no_grad()
    def _match(
        self,
        asso_pred: torch.Tensor,
        asso_gt: torch.Tensor,
        match_cues: torch.Tensor,
        n_t: torch.Tensor,
    ) -> torch.Tensor:
        """Match predicted scores to gt scores using match cues.

        Args:
            asso_pred: Association matrix output from the transformer forward
                pass denoting predicted instances over time (M x N)
            asso_gt: Ground truth association matrix (K x N) denoting ground
                truth instances over time
            match_cues: Tensor indicating which instance is assigned to each gt
                detection (K x 3) or (N,)
            n_t: number of ground truth instances (N,)

        Returns:
            src_inds: Matched source indices (N,)
            target_inds: Matched target indices (N,)
        """
        src_inds = torch.where(match_cues >= 0)[0]
        target_inds = match_cues[src_inds]

        return (src_inds, target_inds)
