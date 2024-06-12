"""Helper functions for calculating mot metrics."""

import numpy as np
import motmetrics as mm
import torch
from typing import Iterable
import pandas as pd

# from dreem.inference.post_processing import _pairwise_iou
# from dreem.inference.boxes import Boxes


def get_matches(frames: list["dreem.io.Frame"]) -> tuple[dict, list, int]:
    """Get comparison between predicted and gt trajectory labels.

    Args:
        frames: a list of Frames containing the video_id, frame_id,
            gt labels and predicted labels

    Returns:
        matches: a dict containing predicted and gt trajectory labels
        indices: the frame indices being compared
        video_id: the video being
    """
    matches = {}
    indices = []

    video_id = frames[0].video_id.item()

    if any([frame.has_instances() for frame in frames]):
        for idx, frame in enumerate(frames):
            indices.append(frame.frame_id.item())
            for gt_track_id, pred_track_id in zip(
                frame.get_gt_track_ids(), frame.get_pred_track_ids()
            ):
                match = f"{gt_track_id} -> {pred_track_id}"

                if match not in matches:
                    matches[match] = np.full(len(frames), 0)

                matches[match][idx] = 1
    # else:
    #     warnings.warn("No instances detected!")
    return matches, indices, video_id


def get_switches(matches: dict, indices: list) -> dict:
    """Get misassigned predicted trajectory labels.

    Args:
        matches: a dict containing the gt and predicted labels
        indices: a list of frame indices being used

    Returns:
        A dict of dicts containing the frame at which the switch occured
        and the change in labels
    """
    track, switches = {}, {}
    if len(matches) > 0 and len(indices) > 0:
        matches_key = np.array(list(matches.keys()))
        matches = np.array(list(matches.values()))
        num_frames = matches.shape[1]

        assert num_frames == len(indices)

        for i, idx in zip(range(num_frames), indices):
            switches[idx] = {}

            col = matches[:, i]
            match_indices = np.where(col == 1)[0]
            match_i = [
                (m.split(" ")[0], m.split(" ")[-1]) for m in matches_key[match_indices]
            ]

            for m in match_i:
                gt, pred = m

                if gt in track and track[gt] != pred:
                    switches[idx][gt] = {
                        "frames": (idx - 1, idx),
                        "pred tracks (from, to)": (track[gt], pred),
                    }

                track[gt] = pred

    return switches


def get_switch_count(switches: dict) -> int:
    """Get the number of mislabeled predicted trajectories.

    Args:
        switches: a dict of dicts containing the mislabeled trajectories
            and the frames at which they occur

    Returns:
        the number of switched labels in the video chunk
    """
    only_switches = {k: v for k, v in switches.items() if v != {}}
    sw_cnt = sum([len(i) for i in list(only_switches.values())])
    return sw_cnt


def to_track_eval(frames: list["dreem.io.Frame"]) -> dict:
    """Reformats frames the output from `sliding_inference` to be used by `TrackEval`.

    Args:
        frames: A list of Frames. `See dreem.io.data_structures for more info`.

    Returns:
        data: A dictionary. Example provided below.

    # --------------------------- An example of data --------------------------- #

    *: number of ids for gt at every frame of the video
    ^: number of ids for tracker at every frame of the video
    L: length of video

    data = {
        "num_gt_ids": total number of unique gt ids,
        "num_tracker_dets": total number of detections by your detection algorithm,
        "num_gt_dets": total number of gt detections,
        "gt_ids": (L, *),  # Ragged np.array
        "tracker_ids": (L, ^),  # Ragged np.array
        "similarity_scores": (L, *, ^),  # Ragged np.array
        "num_timesteps": L,
    }
    """
    unique_gt_ids = []
    num_tracker_dets = 0
    num_gt_dets = 0
    gt_ids = []
    track_ids = []
    similarity_scores = []

    data = {}
    cos_sim = torch.nn.CosineSimilarity()

    for fidx, frame in enumerate(frames):
        gt_track_ids = frame.get_gt_track_ids().cpu().numpy().tolist()
        pred_track_ids = frame.get_pred_track_ids().cpu().numpy().tolist()
        # boxes = Boxes(frame.get_bboxes().cpu())

        gt_ids.append(np.array(gt_track_ids))
        track_ids.append(np.array(pred_track_ids))

        num_tracker_dets += len(pred_track_ids)
        num_gt_dets += len(gt_track_ids)

        if not set(gt_track_ids).issubset(set(unique_gt_ids)):
            unique_gt_ids.extend(list(set(gt_track_ids).difference(set(unique_gt_ids))))

        # eval_matrix = _pairwise_iou(boxes, boxes)
        eval_matrix = np.full((len(gt_track_ids), len(pred_track_ids)), np.nan)

        for i, feature_i in enumerate(frame.get_features()):
            for j, feature_j in enumerate(frame.get_features()):
                eval_matrix[i][j] = cos_sim(
                    feature_i.unsqueeze(0), feature_j.unsqueeze(0)
                )

        # eval_matrix
        #                      pred_track_ids
        #                            0        1
        #  gt_track_ids    1        ...      ...
        #                  0        ...      ...
        #
        # Since the order of both gt_track_ids and pred_track_ids matter (maps from pred to gt),
        # we know the diagonal is the important part. E.g. gt_track_ids=1 maps to pred_track_ids=0
        # and gt_track_ids=0 maps to pred_track_ids=1 because they are ordered in that way.

        # Based on assumption that eval_matrix is always a square matrix.
        # This is true because we are using ground-truth detections.
        #
        # - The number of predicted tracks for a frame will always be the same number
        # of ground truth tracks for a frame.
        # - The number of predicted and ground truth detections will always be the same
        # for any frame.
        # - Because we map detections to features one-to-one, there will always be the same
        # number of features for both predicted and ground truth for any frame.

        # Mask upper and lower triangles of the square matrix (set to 0).
        eval_matrix = np.triu(np.tril(eval_matrix))

        # Replace the 0s with np.nans.
        i, j = np.where(eval_matrix == 0)
        eval_matrix[i, j] = np.nan

        similarity_scores.append(eval_matrix)

    data["num_gt_ids"] = len(unique_gt_ids)
    data["num_tracker_dets"] = num_tracker_dets
    data["num_gt_dets"] = num_gt_dets
    try:
        data["gt_ids"] = gt_ids
        # print(data['gt_ids'])
    except Exception as e:
        print(gt_ids)
        raise (e)
    data["tracker_ids"] = track_ids
    data["similarity_scores"] = similarity_scores
    data["num_timesteps"] = len(frames)

    return data


def get_track_evals(data: dict, metrics: dict) -> dict:
    """Run track_eval and get mot metrics.

    Args:
        data: A dictionary. Example provided below.
        metrics: mot metrics to be computed
    Returns:
        A dictionary with key being the metric, and value being the metric value computed.
    # --------------------------- An example of data --------------------------- #

    *: number of ids for gt at every frame of the video
    ^: number of ids for tracker at every frame of the video
    L: length of video

    data = {
        "num_gt_ids": total number of unique gt ids,
        "num_tracker_dets": total number of detections by your detection algorithm,
        "num_gt_dets": total number of gt detections,
        "gt_ids": (L, *),  # Ragged np.array
        "tracker_ids": (L, ^),  # Ragged np.array
        "similarity_scores": (L, *, ^),  # Ragged np.array
        "num_timsteps": L,
    }
    """
    results = {}
    for metric_name, metric in metrics.items():
        result = metric.eval_sequence(data)
        results.update(result)
    return results


def get_pymotmetrics(
    data: dict,
    metrics: str | tuple = "all",
    key: str = "tracker_ids",
    save: str | None = None,
) -> pd.DataFrame:
    """Given data and a key, evaluate the predictions.

    Args:
        data: A dictionary. Example provided below.
        key: The key within instances to look for track_ids (can be "gt_ids" or "tracker_ids").

    Returns:
        summary: A pandas DataFrame of all the pymot-metrics.

    # --------------------------- An example of data --------------------------- #

    *: number of ids for gt at every frame of the video
    ^: number of ids for tracker at every frame of the video
    L: length of video

    data = {
        "num_gt_ids": total number of unique gt ids,
        "num_tracker_dets": total number of detections by your detection algorithm,
        "num_gt_dets": total number of gt detections,
        "gt_ids": (L, *),  # Ragged np.array
        "tracker_ids": (L, ^),  # Ragged np.array
        "similarity_scores": (L, *, ^),  # Ragged np.array
        "num_timsteps": L,
    }
    """
    if not isinstance(metrics, str):
        metrics = [
            "num_switches" if metric.lower() == "sw_cnt" else metric
            for metric in metrics
        ]  # backward compatibility
    acc = mm.MOTAccumulator(auto_id=True)

    for i in range(len(data["gt_ids"])):
        acc.update(
            oids=data["gt_ids"][i],
            hids=data[key][i],
            dists=data["similarity_scores"][i],
        )

    mh = mm.metrics.create()

    all_metrics = [
        metric.split("|")[0] for metric in mh.list_metrics_markdown().split("\n")[2:-1]
    ]

    if isinstance(metrics, str):
        metrics_list = all_metrics

    elif isinstance(metrics, Iterable):
        metrics = [metric.lower() for metric in metrics]
        metrics_list = [metric for metric in all_metrics if metric.lower() in metrics]

    else:
        raise TypeError(
            f"Metrics must either be an iterable of strings or `all` not: {type(metrics)}"
        )

    summary = mh.compute(acc, metrics=metrics_list, name="acc")
    summary = summary.transpose()

    if save is not None and save != "":
        summary.to_csv(save)

    return summary["acc"]
