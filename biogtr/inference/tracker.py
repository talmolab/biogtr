"""Module containing logic for going from association -> assignment."""
import torch
import pandas as pd
from biogtr.models import model_utils
from biogtr.inference import post_processing
from biogtr.inference.boxes import Boxes
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from scipy.optimize import linear_sum_assignment
from copy import deepcopy


class Tracker:
    """Tracker class used for assignment based on sliding inference from GTR."""

    def __init__(
        self,
        model: GlobalTrackingTransformer,
        window_size: int = 8,
        use_vis_feats: bool = True,
        overlap_thresh: float = 0.01,
        mult_thresh: bool = True,
        decay_time: float = None,
        iou: str = None,
        max_center_dist: float = None,
    ):
        """Initialize a tracker to run inference.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            window_size: the size of the window used during sliding inference
            use_vis_feats: Whether or not to use visual feature extractor
            overlap_thresh: the trajectory overlap threshold to be used for assignment
            mult_thresh: Whether or not to use weight threshold
            decay_time: weight for `decay_time` postprocessing
            iou: Either [None, '', "mult" or "max"]
                 Whether to use multiplicative or max iou reweighting
            max_center_dist: distance threshold for filtering trajectory score matrix
        """
        self.model = model
        _ = self.model.eval()
        self.window_size = window_size
        self.use_vis_feats = use_vis_feats
        self.overlap_thresh = overlap_thresh
        self.mult_thresh = mult_thresh
        self.decay_time = decay_time
        self.iou = iou
        self.max_center_dist = max_center_dist

    def __call__(self, instances: list[dict], all_instances: list = None):
        """Wrapper around `track` to enable `tracker()` instead of `tracker.track()`.

        Args:
            instances: data dict to run inference on
            all_instances: list of instances from previous chunks
                           to stitch together full trajectory

        Returns:
            instances dict populated with pred track ids and association matrix scores
        """
        return self.track(instances, all_instances)

    def track(self, instances: list[dict], all_instances: list = None):
        """Run tracker and get predicted trajectories.

        Args:
            instances: data dict to run inference on
            all_instances: list of instances from previous chunks to stitch together full trajectory

        Returns:
            instances dict populated with pred track ids and association matrix scores
        """
        # Extract feature representations with pre-trained encoder.
        for frame in instances:
            if (frame["num_detected"] > 0).item():
                if not self.use_vis_feats:
                    num_frame_instances = frame["crops"].shape[0]
                    frame["features"] = torch.zeros(
                        num_frame_instances, self.model.d_model
                    )
                    # frame["features"] = torch.randn(
                    #     num_frame_instances, self.model.d_model
                    # )

                # comment out to turn encoder off

                # Assuming the encoder is already trained or train encoder jointly.
                else:
                    with torch.no_grad():
                        z = self.model.visual_encoder(frame["crops"])
                        frame["features"] = z

        # I feel like this chunk is unnecessary:
        # reid_features = torch.cat(
        #     [frame["features"] for frame in instances], dim=0
        # ).unsqueeze(0)

        # asso_preds, pred_boxes, pred_time, embeddings = self.model(
        #     instances, reid_features
        # )
        return self.sliding_inference(
            instances, window_size=self.window_size, all_instances=all_instances
        )

    def sliding_inference(self, instances, window_size, all_instances=None):
        """Performs sliding inference on the input video (instances) with a given window size.

        Args:
            instances: A list of dictionaries, one dictionary for each frame. An example
            is provided below.
            window_size: An integer.

        Returns:
            instances: A list of dictionaries, one dictionary for each frame. An example
            is provided below.
        # ------------------------- An example of instances ------------------------ #
        D: embedding dimension.
        N_i: number of detected instances in i-th frame of window.
        instances = [
            {
                # Each dictionary is a frame.
                "frame_id": frame index int,
                "num_detected": N_i,
                "gt_track_ids": (N_i,),
                "poses": (N_i, 13, 2),  # 13 keypoints for the pose (x, y) coords.
                "bboxes": (N_i, 4),  # in pascal_voc unrounded unnormalized
                "features": (N_i, D),  # Features are deleted but can optionally be kept if need be.
                "pred_track_ids": (N_i,),  # Filled out after sliding_inference.
            },
            {},  # Frame 2.
            ...
        ]
        """
        # B: batch size.
        # D: embedding dimension.
        # nc: number of channels.
        # H: height.
        # W: width.

        video_len = len(instances)
        id_count = 0

        for frame_id in range(video_len):
            if frame_id == 0:
                if all_instances is not None and len(all_instances) != 0:
                    instances[0]["pred_track_ids"] = torch.arange(
                        0, len(all_instances[-1]["bboxes"])
                    )
                    id_count = len(all_instances[-1]["bboxes"])

                    test = [all_instances[-1], instances[0]]

                    test, id_count = self._run_global_tracker(
                        test,
                        k=1,
                        id_count=id_count,
                        overlap_thresh=self.overlap_thresh,
                        mult_thresh=self.mult_thresh,
                    )

                    instances[0] = test[-1]

                    # print('first frame of new chunk!', instances[frame_id]['pred_track_ids'])
                else:
                    instances[0]["pred_track_ids"] = torch.arange(
                        0, len(instances[0]["bboxes"])
                    )
                    id_count = len(instances[0]["bboxes"])

                    # print('id count: ', id_count)
                    # print('first overall frame!', instances[frame_id]['pred_track_ids'])
            else:
                win_st = max(0, frame_id + 1 - window_size)
                win_ed = frame_id + 1
                instances[win_st:win_ed], id_count = self._run_global_tracker(
                    instances[win_st:win_ed],
                    k=min(window_size - 1, frame_id),
                    id_count=id_count,
                    overlap_thresh=self.overlap_thresh,
                    mult_thresh=self.mult_thresh,
                )
                # print(f'frame: {frame_id}', instances[frame_id]['pred_track_ids'])

            """
            # If first frame.
            if frame_id == 0:
                instances[0]["pred_track_ids"] = torch.arange(
                    0, len(instances[0]["bboxes"]))
                id_count = len(instances[0]["bboxes"])
            else:
                win_st = max(0, frame_id + 1 - window_size)
                win_ed = frame_id + 1
                instances[win_st: win_ed], id_count = self._run_global_tracker(
                    instances[win_st: win_ed],
                    k=min(window_size - 1, frame_id),
                    id_count=id_count,
                    overlap_thresh=self.overlap_thresh,
                    mult_thresh=self.mult_thresh)
            """

            # If features are out of window, set to none.
            # if frame_id - window_size >= 0:
            #     instances[frame_id - window_size]["features"] = None

        # TODO: Insert postprocessing.

        # Remove last few features from cuda.
        for frame in instances[-window_size:]:
            frame["features"] = frame["features"].cpu()

        return instances

    def _run_global_tracker(self, instances, k, id_count, overlap_thresh, mult_thresh):
        """Run_global_tracker performs the actual tracking.

        Uses Hungarian algorithm to do track assigning.

        Args:
            instances: A list of dictionaries, one dictionary for each frame. An example
            is provided below.
            k: An integer for the query frame within the window of instances.
            id_count: The count of total identities so far.
            overlap_thresh: A float number between 0 and 1 specifying how much
            overlap is necessary for assigning a new instance to an existing identity.
            mult_thresh: A boolean for whether or not multiple thresholds should be used.
            This is not functional as of now.

        Returns:
            instances: The exact list of dictionaries as before but with assigned track ids
            and new track ids for the query frame. Refer to the example for the structure.
            id_count: An integer for the updated identity count so far.
            # ------------------------- An example of instances ------------------------ #
            NOTE: This instances variable is the window subset of the instances variable in sliding_inference.
            *: each item in instances is a frame in the window. So it follows
                that each frame in the window has * detected instances.
            D: embedding dimension.
            N_i: number of detected instances in i-th frame of window.
            T: length of window.
            The features in instances can be of shape (2 to T, *, D) when stacked together.
            instances = [
                {
                    # Each dictionary is a frame.
                    "frame_id": frame index int,
                    "num_detected": N_i,
                    "gt_track_ids": (N_i,),
                    "poses": (N_i, 13, 2),  # 13 keypoints for the pose (x, y) coords.
                    "bboxes": (N_i, 4),  # in pascal_voc unrounded unnormalized
                    "features": (N_i, D),
                    "pred_track_ids": (N_i,),  # Before assignnment, these are all -1.
                },
                ...
            ]
        """
        # *: each item in instances is a frame in the window. So it follows
        #    that each frame in the window has * detected instances.
        # D: embedding dimension.
        # N: number of instances in the window.
        # N_i: number of detected instances in i-th frame of window.
        # n_t: a list of number of instances in each frame of the window.
        # N_t: number of instances in current/query frame (rightmost frame of the window).
        # Np: number of instances in the window excluding the current/query frame.
        # T: length of window.
        # L: number of decoder blocks.
        # M: number of existing tracks within the window so far.

        # Number of instances in each frame of the window.
        # E.g.: n_t: [4, 5, 6, 7]; window of length 4 with 4 detected instances in the first frame of the window.
        n_t = [frame["num_detected"] for frame in instances]

        N, T = sum(n_t), len(n_t)  # Number of instances in window; length of window.
        reid_features = torch.cat([frame["features"] for frame in instances], dim=0)[
            None
        ]  # (1, N, D=512)

        # (L=1, N_t, N)
        with torch.no_grad():
            if self.model.transformer.return_embedding:
                asso_output, embed = self.model(instances, query_frame=k)
                instances[k]["embeddings"] = embed
            else:
                asso_output = self.model(instances, query_frame=k)

        asso_output = asso_output[-1].split(n_t, dim=1)  # (T, N_t, N_i)
        asso_output = model_utils.softmax_asso(asso_output)  # (T, N_t, N_i)
        asso_output = torch.cat(asso_output, dim=1).cpu()  # (N_t, N)

        N_t = instances[k][
            "num_detected"
        ]  # Number of instances in the current/query frame.

        N_p = (
            N - N_t
        )  # Number of instances in the window not including the current/query frame.

        ids = torch.cat(
            [x["pred_track_ids"] for t, x in enumerate(instances) if t != k], dim=0
        ).view(
            N_p
        )  # (N_p,)

        k_inds = [x for x in range(sum(n_t[:k]), sum(n_t[: k + 1]))]
        nonk_inds = [i for i in range(N) if i not in k_inds]
        asso_nonk = asso_output[:, nonk_inds]  # (N_t, N_p)

        pred_boxes, _ = model_utils.get_boxes_times(instances)
        pred_boxes = pred_boxes.cpu()

        k_boxes = pred_boxes[k_inds]  # n_k x 4
        nonk_boxes = pred_boxes[nonk_inds]  # Np x 4

        # TODO: Insert postprocessing.

        unique_ids = torch.unique(ids)  # (M,)
        M = len(unique_ids)  # Number of existing tracks.
        id_inds = (unique_ids[None, :] == ids[:, None]).float()  # (N_p, M)

        ################################################################################

        # reweighting hyper-parameters for association -> they use 0.9

        # (n_k x Np) x (Np x M) --> n_k x M
        traj_score = post_processing.weight_decay_time(
            asso_nonk, self.decay_time, reid_features, T, k
        )

        traj_score = torch.mm(traj_score, id_inds.cpu())  # (N_t, M)

        instances[k]["decay_time_traj_score"] = pd.DataFrame(
            traj_score.cpu().numpy(), columns=unique_ids.cpu().numpy()
        )
        instances[k]["decay_time_traj_score"].index.name = "Current Frame Instances"
        instances[k]["decay_time_traj_score"].columns.name = "Unique IDs"
        ################################################################################

        # with iou -> combining with location in tracker, they set to True
        # todo -> should also work without pos_embed

        if id_inds.numel() > 0:
            # this throws error, think we need to slice?
            # last_inds = (id_inds * torch.arange(
            #    N_p, device=id_inds.device)[:, None]).max(dim=0)[1] # M

            last_inds = (
                id_inds * torch.arange(N_p[0], device=id_inds.device)[:, None]
            ).max(dim=0)[
                1
            ]  # M

            last_boxes = nonk_boxes[last_inds]  # M x 4
            last_ious = post_processing._pairwise_iou(
                Boxes(k_boxes), Boxes(last_boxes)
            )  # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)

        traj_score = post_processing.weight_iou(traj_score, self.iou, last_ious.cpu())

        ################################################################################

        # threshold for continuing a tracking or starting a new track -> they use 1.0
        # todo -> should also work without pos_embed
        traj_score = post_processing.filter_max_center_dist(
            traj_score, self.max_center_dist, k_boxes, nonk_boxes, id_inds
        )

        ################################################################################

        match_i, match_j = linear_sum_assignment((-traj_score))

        track_ids = ids.new_full((N_t,), -1)
        for i, j in zip(match_i, match_j):
            # The overlap threshold is multiplied by the number of times the unique track j is matched to an
            # instance out of all instances in the window excluding the current frame.
            #
            # So if this is correct, the threshold is higher for matching an instance from the current frame
            # to an existing track if that track has already been matched several times.
            # So if an existing track in the window has been matched a lot, it gets harder to match to that track.
            thresh = (
                overlap_thresh * id_inds[:, j].sum() if mult_thresh else overlap_thresh
            )
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]

        for i in range(N_t):
            if track_ids[i] < 0:
                track_ids[i] = id_count
                id_count += 1

        instances[k]["matches"] = (match_i, match_j)
        instances[k]["pred_track_ids"] = track_ids
        instances[k]["final_traj_score"] = pd.DataFrame(
            traj_score.cpu().numpy(), columns=unique_ids.cpu().numpy()
        )
        instances[k]["final_traj_score"].index.name = "Current Frame Instances"
        instances[k]["final_traj_score"].columns.name = "Unique IDs"

        return instances, id_count
