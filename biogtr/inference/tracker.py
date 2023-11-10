"""Module containing logic for going from association -> assignment."""
import torch
import pandas as pd
from biogtr.models import model_utils
from biogtr.inference import post_processing
from biogtr.inference.boxes import Boxes
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from collections import deque


class Tracker:
    """Tracker class used for assignment based on sliding inference from GTR."""

    def __init__(
        self,
        window_size: int = 8,
        use_vis_feats: bool = True,
        overlap_thresh: float = 0.01,
        mult_thresh: bool = True,
        decay_time: float = None,
        iou: str = None,
        max_center_dist: float = None,
        persistent_tracking: bool = False
    ):
        """Initialize a tracker to run inference.

        Args:
            window_size: the size of the window used during sliding inference
            use_vis_feats: Whether or not to use visual feature extractor
            overlap_thresh: the trajectory overlap threshold to be used for assignment
            mult_thresh: Whether or not to use weight threshold
            decay_time: weight for `decay_time` postprocessing
            iou: Either [None, '', "mult" or "max"]
                 Whether to use multiplicative or max iou reweighting
            max_center_dist: distance threshold for filtering trajectory score matrix
            persistent_tracking: whether to keep a buffer across chunks or not
        """
        
        self.window_size = window_size
        self.track_queue = deque(maxlen=self.window_size)
        self.use_vis_feats = use_vis_feats
        self.overlap_thresh = overlap_thresh
        self.mult_thresh = mult_thresh
        self.decay_time = decay_time
        self.iou = iou
        self.max_center_dist = max_center_dist
        self.persistent_tracking = persistent_tracking

    def __call__(self, model: GlobalTrackingTransformer, instances: list[dict], all_instances: list = None):
        """Wrapper around `track` to enable `tracker()` instead of `tracker.track()`.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            instances: data dict to run inference on
            all_instances: list of instances from previous chunks
                           to stitch together full trajectory

        Returns:
            instances dict populated with pred track ids and association matrix scores
        """
        return self.track(model, instances, all_instances)

    def track(self, model: GlobalTrackingTransformer, instances: list[dict], all_instances: list = None):
        """Run tracker and get predicted trajectories.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
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
                        num_frame_instances, model.d_model
                    )
                    # frame["features"] = torch.randn(
                    #     num_frame_instances, self.model.d_model
                    # )

                # comment out to turn encoder off

                # Assuming the encoder is already trained or train encoder jointly.
                elif 'features' not in frame or frame['features'] == None:
                    with torch.no_grad():
                        z = model.visual_encoder(frame["crops"])
                        frame["features"] = z

        # I feel like this chunk is unnecessary:
        # reid_features = torch.cat(
        #     [frame["features"] for frame in instances], dim=0
        # ).unsqueeze(0)

        # asso_preds, pred_boxes, pred_time, embeddings = self.model(
        #     instances, reid_features
        # )
        instances_pred = self.sliding_inference(
            model, instances, window_size=self.window_size, all_instances=all_instances
        )

        if not self.persistent_tracking:
            self.track_queue.clear()
            
        return instances_pred

    def sliding_inference(self, model: GlobalTrackingTransformer, instances, window_size, all_instances=None):
        """Performs sliding inference on the input video (instances) with a given window size.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
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

        for batch_idx in range(video_len):
            if instances[batch_idx]['frame_id'] == 0 or (not self.persistent_tracking and batch_idx == 0):

                self.track_queue.clear()
                
                instances[batch_idx]["pred_track_ids"] = torch.arange(
                    0, len(instances[0]["bboxes"])
                )

                id_count = len(instances[0]["bboxes"])

                self.track_queue.append(instances[batch_idx])

            else:
                instances_to_track = (list(self.track_queue) + [instances[batch_idx]])[-window_size:]

                if not self.persistent_tracking:
                    query_ind = min(window_size - 1, batch_idx)
                else:
                    query_ind = min(window_size - 1, instances[batch_idx]['frame_id'])
                    
                instances[batch_idx], id_count = self._run_global_tracker(
                    model,
                    instances_to_track,
                    query_frame=query_ind,
                    id_count=id_count,
                    overlap_thresh=self.overlap_thresh,
                    mult_thresh=self.mult_thresh,
                )

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
                    query_frame=min(window_size - 1, frame_id),
                    id_count=id_count,
                    overlap_thresh=self.overlap_thresh,
                    mult_thresh=self.mult_thresh)
            """

            # If features are out of window, set to none.
            # if frame_id - window_size >= 0:
            #     instances[frame_id - window_size]["features"] = None

        # TODO: Insert postprocessing.

        for frame in instances[:len(instances)-window_size]:
            frame["features"] = frame["features"].cpu()

        return instances

    def _run_global_tracker(self, model: GlobalTrackingTransformer, instances, query_frame, id_count, overlap_thresh, mult_thresh):
        """Run_global_tracker performs the actual tracking.

        Uses Hungarian algorithm to do track assigning.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            instances: A list of dictionaries, one dictionary for each frame. An example
            is provided below.
            query_frame: An integer for the query frame within the window of instances.
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
            window_size: length of window.
            The features in instances can be of shape (2 to window_size, *, D) when stacked together.
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
        # total_instances: number of instances in the window.
        # N_i: number of detected instances in i-th frame of window.
        # instances_per_frame: a list of number of instances in each frame of the window.
        # n_query: number of instances in current/query frame (rightmost frame of the window).
        # n_nonquery: number of instances in the window excluding the current/query frame.
        # window_size: length of window.
        # L: number of decoder blocks.
        # n_traj: number of existing tracks within the window so far.

        # Number of instances in each frame of the window.
        # E.g.: instances_per_frame: [4, 5, 6, 7]; window of length 4 with 4 detected instances in the first frame of the window.
        _ = model.eval()

        instances_per_frame = [frame["num_detected"] for frame in instances]

        total_instances, window_size = sum(instances_per_frame), len(instances_per_frame)  # Number of instances in window; length of window.
        reid_features = torch.cat([frame["features"] for frame in instances], dim=0)[
            None
        ]  # (1, total_instances, D=512)

        # (L=1, n_query, total_instances)
        with torch.no_grad():
            if model.transformer.return_embedding:
                asso_output, embed = model(instances, query_frame=query_frame)
                instances[query_frame]["embeddings"] = embed
            else:
                asso_output = model(instances, query_frame=query_frame)
        
        asso_output = asso_output[-1].split(instances_per_frame, dim=1)  # (window_size, n_query, N_i)
        asso_output = model_utils.softmax_asso(asso_output)  # (window_size, n_query, N_i)
        asso_output = torch.cat(asso_output, dim=1).cpu()  # (n_query, total_instances)

        n_query = instances[query_frame][
            "num_detected"
        ]  # Number of instances in the current/query frame.

        n_nonquery = (
            total_instances - n_query
        )  # Number of instances in the window not including the current/query frame.

        instance_ids = torch.cat(
            [x["pred_track_ids"] for batch_idx, x in enumerate(instances) if batch_idx != query_frame], dim=0
        ).view(
            n_nonquery
        )  # (n_nonquery,)

        query_inds = [x for x in range(sum(instances_per_frame[:query_frame]), sum(instances_per_frame[: query_frame + 1]))]
        nonquery_inds = [i for i in range(total_instances) if i not in query_inds]
        asso_nonquery = asso_output[:, nonquery_inds]  # (n_query, n_nonquery)

        pred_boxes, _ = model_utils.get_boxes_times(instances)
        query_boxes = pred_boxes[query_inds]  # n_k x 4
        nonquery_boxes = pred_boxes[nonquery_inds]  #n_nonquery x 4
        # TODO: Insert postprocessing.

        unique_ids = torch.unique(instance_ids)  # (n_nonquery,)
        n_traj = len(unique_ids)  # Number of existing tracks.
        id_inds = (unique_ids[None, :] == instance_ids[:, None]).float()  # (n_nonquery, n_traj)

        ################################################################################

        # reweighting hyper-parameters for association -> they use 0.9

        # (n_query x n_nonquery) x (n_nonquery x n_traj) --> n_k x n_traj
        traj_score = post_processing.weight_decay_time(
            asso_nonquery, self.decay_time, reid_features, window_size, query_frame
        )

        traj_score = torch.mm(traj_score, id_inds.cpu())  # (n_query, n_traj)

        instances[query_frame]["decay_time_traj_score"] = pd.DataFrame(
            deepcopy((traj_score).numpy()), columns=unique_ids.cpu().numpy()
        )
        instances[query_frame]["decay_time_traj_score"].index.name = "Current Frame Instances"
        instances[query_frame]["decay_time_traj_score"].columns.name = "Unique IDs"
        ################################################################################

        # with iou -> combining with location in tracker, they set to True
        # todo -> should also work without pos_embed

        if id_inds.numel() > 0:
            # this throws error, think we need to slice?
            # last_inds = (id_inds * torch.arange(
            #    n_nonquery, device=id_inds.device)[:, None]).max(dim=0)[1] # n_traj

            nonquery_inds = (
                id_inds * torch.arange(n_nonquery[0], device=id_inds.device)[:, None]
            ).max(dim=0)[
                1
            ]  # M

            nonquery_boxes = nonquery_boxes[nonquery_inds]  # n_traj x 4
            nonquery_ious = post_processing._pairwise_iou(
                Boxes(query_boxes), Boxes(nonquery_boxes)
            )  # n_k x M
        else:
            nonquery_ious = traj_score.new_zeros(traj_score.shape)
        traj_score = post_processing.weight_iou(traj_score, self.iou, nonquery_ious.cpu())
        ################################################################################

        # threshold for continuing a tracking or starting a new track -> they use 1.0
        # todo -> should also work without pos_embed
        traj_score = post_processing.filter_max_center_dist(
            traj_score, self.max_center_dist, query_boxes, nonquery_boxes, id_inds
        )

        ################################################################################

        match_i, match_j = linear_sum_assignment((-traj_score))

        track_ids = instance_ids.new_full((n_query,), -1)
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

        for i in range(n_query):
            if track_ids[i] < 0:
                track_ids[i] = id_count
                id_count += 1

        instances[query_frame]["matches"] = (match_i, match_j)
        instances[query_frame]["pred_track_ids"] = track_ids
        instances[query_frame]["final_traj_score"] = pd.DataFrame(
            deepcopy((traj_score).numpy()), columns=unique_ids.cpu().numpy()
        )
        instances[query_frame]["final_traj_score"].index.name = "Current Frame Instances"
        instances[query_frame]["final_traj_score"].columns.name = "Unique IDs"

        self.track_queue.append(instances[query_frame])

        return instances[query_frame], id_count
