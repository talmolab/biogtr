"""Test inference logic."""
import torch
import pytest
import numpy as np
from biogtr.data_structures import Frame, Instance
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.inference.tracker import Tracker
from biogtr.inference import post_processing
from biogtr.inference import metrics


def test_tracker():
    """Test tracker module.

    Tests that tracker works with/without post processing
    """
    feats = 512
    num_frames = 5
    num_detected = 2
    img_shape = (1, 128, 128)
    test_frame = 1
    frames = []

    for i in range(num_frames):
        instances = []
        for j in range(num_detected):
            instances.append(
                Instance(
                    gt_track_id=j,
                    pred_track_id=-1,
                    bbox=torch.rand(size=(1, 4)),
                    crop=torch.rand(size=(1, 1, 64, 64)),
                )
            )
        frames.append(
            Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        )

    embedding_meta = {
        "embedding_type": "fixed_pos",
        "kwargs": {"temperature": num_detected, "scale": num_frames, "normalize": True},
    }

    tracking_transformer = GlobalTrackingTransformer(
        encoder_model="resnet18",
        encoder_cfg={"weights": "ResNet18_Weights.DEFAULT"},
        d_model=feats,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=feats,
        feature_dim_attn_head=feats,
        embedding_meta=embedding_meta,
        return_embedding=False,
    )

    tracking_cfg = {
        "overlap_thresh": 0.01,
        "decay_time": None,
        "iou": None,
        "max_center_dist": None,
    }

    tracker = Tracker(**tracking_cfg)

    frames_pred = tracker(tracking_transformer, frames)

    # TODO: Debug saving asso matrices
    # asso_equals = (
    #     frames_pred[test_frame].get_traj_score("decay_time").to_numpy()
    #     == frames_pred[test_frame].get_traj_score("final").to_numpy()
    # ).all()
    # assert asso_equals

    assert len(frames_pred[test_frame].get_pred_track_ids()) == num_detected


# @pytest.mark.parametrize("set_default_device", ["cpu"], indirect=True)
def test_post_processing():  # set_default_device
    """Test postprocessing methods.

    Tests each postprocessing method to ensure that
    it filters only when the condition is satisfied.
    """
    T = 8
    k = 5
    D = 512
    M = 5
    N_t = 5
    N_p = N_t * (T - 1)
    N = N_t * T

    reid_features = torch.rand((1, N_t, D))
    asso_nonk = torch.rand((N_t, N_p))

    decay_time = 0

    assert (
        asso_nonk
        == post_processing.weight_decay_time(asso_nonk, decay_time, reid_features, T, k)
    ).all()

    decay_time = 0.9

    assert not (
        asso_nonk
        == post_processing.weight_decay_time(asso_nonk, decay_time, reid_features, T, k)
    ).all()

    asso_output = torch.rand((N_t, M))
    ious = torch.rand((N_t, M))

    assert (asso_output == post_processing.weight_iou(asso_output, None, ious)).all()

    assert not (
        asso_output == post_processing.weight_iou(asso_output, "mult", ious)
    ).all()

    assert not (
        asso_output == post_processing.weight_iou(asso_output, "max", ious)
    ).all()

    assert not (
        post_processing.weight_iou(asso_output, "mult", ious)
        == post_processing.weight_iou(asso_output, "max", ious)
    ).all()

    im_size = 128
    k_boxes = torch.rand((N_t, 4)) * im_size
    nonk_boxes = torch.rand((N_p, 4)) * im_size
    id_inds = torch.tile(torch.cat((torch.zeros(M - 1), torch.ones(1))), (N_p, 1))

    assert (
        asso_output
        == post_processing.filter_max_center_dist(
            asso_output=asso_output,
            max_center_dist=0,
            k_boxes=k_boxes,
            nonk_boxes=nonk_boxes,
            id_inds=id_inds,
        )
    ).all()

    assert not (
        asso_output
        == post_processing.filter_max_center_dist(
            asso_output=asso_output,
            max_center_dist=1e-9,
            k_boxes=k_boxes,
            nonk_boxes=nonk_boxes,
            id_inds=id_inds,
        )
    ).all()


def test_metrics():
    """Test basic GTR Runner."""
    num_frames = 3
    num_detected = 3
    n_batches = 1
    batches = []

    for i in range(n_batches):
        frames_pred = []
        for j in range(num_frames):
            instances_pred = []
            for k in range(num_detected):
                bboxes = torch.tensor(np.random.uniform(size=(num_detected, 4)))
                bboxes[:, -2:] += 1
                instances_pred.append(
                    Instance(gt_track_id=k, pred_track_id=k, bbox=torch.randn((1, 4)))
                )
            frames_pred.append(Frame(video_id=0, frame_id=j, instances=instances_pred))
        batches.append(frames_pred)

    for batch in batches:
        instances_mm = metrics.to_track_eval(batch)
        clear_mot = metrics.get_pymotmetrics(instances_mm)

        matches, indices, _ = metrics.get_matches(batch)

        switches = metrics.get_switches(matches, indices)

        sw_cnt = metrics.get_switch_count(switches)

        assert sw_cnt == clear_mot["num_switches"] == 0, (
            sw_cnt,
            clear_mot["num_switches"],
        )
