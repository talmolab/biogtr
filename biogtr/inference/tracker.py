"""Module containing logic for going from association -> assignment."""
import torch
import pandas as pd
import warnings
from biogtr.data_structures import Frame
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.models import model_utils
from biogtr.inference.track_queue import TrackQueue
from biogtr.inference import post_processing
from biogtr.inference.boxes import Boxes
from scipy.optimize import linear_sum_assignment
from copy import deepcopy


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
        persistent_tracking: bool = False,
        max_gap: int = -1,
        verbose=False,
    ):
        """Initialize a tracker to run inference.

        Args:
            window_size: the size of the window used during sliding inference.
            use_vis_feats: Whether or not to use visual feature extractor.
            overlap_thresh: the trajectory overlap threshold to be used for assignment.
            mult_thresh: Whether or not to use weight threshold.
            decay_time: weight for `decay_time` postprocessing.
            iou: Either [None, '', "mult" or "max"]
                 Whether to use multiplicative or max iou reweighting.
            max_center_dist: distance threshold for filtering trajectory score matrix.
            persistent_tracking: whether to keep a buffer across chunks or not.
            max_gap: the max number of frames a trajectory can be missing before termination.
            verbose: Whether or not to turn on debug printing after each operation.
        """
        self.track_queue = TrackQueue(
            window_size=window_size, max_gap=max_gap, verbose=verbose
        )
        self.use_vis_feats = use_vis_feats
        self.overlap_thresh = overlap_thresh
        self.mult_thresh = mult_thresh
        self.decay_time = decay_time
        self.iou = iou
        self.max_center_dist = max_center_dist
        self.persistent_tracking = persistent_tracking
        self.verbose = verbose

    def __call__(self, model: GlobalTrackingTransformer, frames: list[Frame]):
        """Wrap around `track` to enable `tracker()` instead of `tracker.track()`.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: list of Frames to run inference on

        Returns:
            List of frames containing association matrix scores and instances populated with pred track ids.
        """
        return self.track(model, frames)

    def track(self, model: GlobalTrackingTransformer, frames: list[dict]):
        """Run tracker and get predicted trajectories.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: data dict to run inference on

        Returns:
            List of Frames populated with pred track ids and association matrix scores
        """
        # Extract feature representations with pre-trained encoder.

        _ = model.eval()

        for frame in frames:
            if frame.has_instances():
                if not self.use_vis_feats:
                    for instance in frame.instances:
                        instance.features = torch.zeros(1, model.d_model)
                    # frame["features"] = torch.randn(
                    #     num_frame_instances, self.model.d_model
                    # )

                # comment out to turn encoder off

                # Assuming the encoder is already trained or train encoder jointly.
                elif not frame.has_features():
                    with torch.no_grad():
                        crops = frame.get_crops()
                        z = model.visual_encoder(crops)

                        for i, z_i in enumerate(z):
                            frame.instances[i].features = z_i

        # I feel like this chunk is unnecessary:
        # reid_features = torch.cat(
        #     [frame["features"] for frame in instances], dim=0
        # ).unsqueeze(0)

        # asso_preds, pred_boxes, pred_time, embeddings = self.model(
        #     instances, reid_features
        # )
        instances_pred = self.sliding_inference(model, frames)

        if not self.persistent_tracking:
            if self.verbose:
                warnings.warn(f"Clearing Queue after tracking")
            self.track_queue.end_tracks()

        return instances_pred

    def sliding_inference(self, model: GlobalTrackingTransformer, frames: list[Frame]):
        """Perform sliding inference on the input video (instances) with a given window size.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frame: A list of Frames (See `biogtr.data_structures.Frame` for more info).


        Returns:
            Frames: A list of Frames populated with pred_track_ids and asso_matrices
        """
        # B: batch size.
        # D: embedding dimension.
        # nc: number of channels.
        # H: height.
        # W: width.

        for batch_idx, frame_to_track in enumerate(frames):
            tracked_frames = self.track_queue.collate_tracks()
            if self.verbose:
                warnings.warn(
                    f"Current number of tracks is {self.track_queue.n_tracks}"
                )

            if (
                self.persistent_tracking and frame_to_track.frame_id == 0
            ):  # check for new video and clear queue
                if self.verbose:
                    warnings.warn("New Video! Resetting Track Queue.")
                self.track_queue.end_tracks()

            """
            Initialize tracks on first frame of video or first instance of detections.
            """
            if len(self.track_queue) == 0:
                if frame_to_track.has_instances():
                    if self.verbose:
                        warnings.warn(
                            f"Initializing track on clip ind {batch_idx} frame {frame_to_track.frame_id.item()}"
                        )

                    for i, instance in enumerate(frames[batch_idx].instances):
                        instance.pred_track_id = i

            else:
                if (
                    frame_to_track.has_instances()
                ):  # Check if there are detections. If there are skip and increment gap count
                    frames_to_track = tracked_frames + [
                        frame_to_track
                    ]  # better var name?

                    query_ind = len(frames_to_track) - 1

                    frame_to_track = self._run_global_tracker(
                        model,
                        frames_to_track,
                        query_ind=query_ind,
                    )

            if frame_to_track.has_instances():
                self.track_queue.add_frame(frame_to_track)
            else:
                self.track_queue.increment_gaps([])

            frames[batch_idx] = frame_to_track
        return frames

    def _run_global_tracker(
        self, model: GlobalTrackingTransformer, frames: list[Frame], query_ind: int
    ) -> Frame:
        """Run global tracker performs the actual tracking.

        Uses Hungarian algorithm to do track assigning.

        Args:
            model: the pretrained GlobalTrackingTransformer to be used for inference
            frames: A list of Frames containing reid features. See `biogtr.data_structures` for more info.
            query_ind: An integer for the query frame within the window of instances.

        Returns:
            query_frame: The query frame now populated with the pred_track_ids.
        """
        # *: each item in frames is a frame in the window. So it follows
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
        query_frame = frames[query_ind]
        instances_per_frame = [frame.num_detected for frame in frames]

        total_instances, window_size = sum(instances_per_frame), len(
            instances_per_frame
        )  # Number of instances in window; length of window.

        overlap_thresh = self.overlap_thresh
        mult_thresh = self.mult_thresh
        n_traj = self.track_queue.n_tracks

        reid_features = torch.cat([frame.get_features() for frame in frames], dim=0)[
            None
        ]  # (1, total_instances, D=512)

        # (L=1, n_query, total_instances)
        with torch.no_grad():
            asso_output, embed = model(frames, query_frame=query_ind)
            # if model.transformer.return_embedding:
            # query_frame.embeddings = embed TODO add embedding to Instance Object
        # if query_frame == 1:
        #     print(asso_output)
        asso_output = asso_output[-1].split(
            instances_per_frame, dim=1
        )  # (window_size, n_query, N_i)
        asso_output = model_utils.softmax_asso(
            asso_output
        )  # (window_size, n_query, N_i)
        asso_output = torch.cat(asso_output, dim=1).cpu()  # (n_query, total_instances)

        try:
            n_query = (
                query_frame.num_detected
            )  # Number of instances in the current/query frame.
        except Exception as e:
            print(len(frames), query_frame, frames[-1])
            raise (e)

        n_nonquery = (
            total_instances - n_query
        )  # Number of instances in the window not including the current/query frame.

        try:
            instance_ids = torch.cat(
                [
                    x.get_pred_track_ids()
                    for batch_idx, x in enumerate(frames)
                    if batch_idx != query_ind
                ],
                dim=0,
            ).view(
                n_nonquery
            )  # (n_nonquery,)
        except Exception as e:
            print(
                [
                    [instance.pred_track_id.device for instance in frame.instances]
                    for frame in frames
                ]
            )
            raise (e)

        query_inds = [
            x
            for x in range(
                sum(instances_per_frame[:query_ind]),
                sum(instances_per_frame[: query_ind + 1]),
            )
        ]
        nonquery_inds = [i for i in range(total_instances) if i not in query_inds]
        asso_nonquery = asso_output[:, nonquery_inds]  # (n_query, n_nonquery)

        pred_boxes, _ = model_utils.get_boxes_times(frames)
        query_boxes = pred_boxes[query_inds]  # n_k x 4
        nonquery_boxes = pred_boxes[nonquery_inds]  # n_nonquery x 4
        # TODO: Insert postprocessing.

        unique_ids = torch.tensor(
            [self.track_queue.tracks], device=instance_ids.device
        ).view(
            n_traj
        )  # (n_nonquery,)

        id_inds = (
            unique_ids[None, :] == instance_ids[:, None]
        ).float()  # (n_nonquery, n_traj)

        ################################################################################

        # reweighting hyper-parameters for association -> they use 0.9

        # (n_query x n_nonquery) x (n_nonquery x n_traj) --> n_k x n_traj
        traj_score = post_processing.weight_decay_time(
            asso_nonquery, self.decay_time, reid_features, window_size, query_ind
        )

        traj_score = torch.mm(traj_score, id_inds.cpu())  # (n_query, n_traj)

        decay_time_traj_score = pd.DataFrame(
            deepcopy((traj_score).numpy()), columns=unique_ids.cpu().numpy()
        )

        decay_time_traj_score.index.name = "Current Frame Instances"
        decay_time_traj_score.columns.name = "Unique IDs"

        query_frame.add_traj_score("decay_time", decay_time_traj_score)
        ################################################################################

        # with iou -> combining with location in tracker, they set to True
        # todo -> should also work without pos_embed

        if id_inds.numel() > 0:
            # this throws error, think we need to slice?
            # last_inds = (id_inds * torch.arange(
            #    n_nonquery, device=id_inds.device)[:, None]).max(dim=0)[1] # n_traj

            last_inds = (
                id_inds * torch.arange(n_nonquery, device=id_inds.device)[:, None]
            ).max(dim=0)[
                1
            ]  # M

            last_boxes = nonquery_boxes[last_inds]  # n_traj x 4
            last_ious = post_processing._pairwise_iou(
                Boxes(query_boxes), Boxes(last_boxes)
            )  # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)
        traj_score = post_processing.weight_iou(traj_score, self.iou, last_ious.cpu())
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
                track_ids[i] = n_traj
                n_traj += 1

        query_frame.matches = (match_i, match_j)

        for instance, track_id in zip(query_frame.instances, track_ids):
            instance.pred_track_id = track_id

        final_traj_score = pd.DataFrame(
            deepcopy((traj_score).numpy()), columns=unique_ids.cpu().numpy()
        )
        final_traj_score.index.name = "Current Frame Instances"
        final_traj_score.columns.name = "Unique IDs"

        query_frame.add_traj_score("final", final_traj_score)
        return query_frame
