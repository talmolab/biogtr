"""Test inference logic."""
import torch
import pytest
from biogtr.models.global_tracking_transformer import GlobalTrackingTransformer
from biogtr.inference.tracker import Tracker
from biogtr.inference import post_processing


def test_tracker():
    """Test tracker module.

    Tests that tracker works with/without post processing
    """
    feats = 512
    num_frames = 5
    num_detected = 2
    img_shape = (1, 128, 128)
    test_frame = 1
    instances = []

    for i in range(num_frames):
        instances.append(
            {
                "frame_id": torch.tensor(i),
                "img_shape": torch.tensor(img_shape),
                "num_detected": torch.tensor([num_detected]),
                "crops": torch.rand(size=(num_detected, 1, 64, 64)),
                "bboxes": torch.rand(size=(num_detected, 4)),
                "gt_track_ids": torch.arange(num_detected),
                "pred_track_ids": torch.tensor([-1] * num_detected),
            }
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

    tracker = Tracker(model=tracking_transformer, **tracking_cfg)

    instances_pred = tracker(instances)

    asso_equals = (
        instances_pred[test_frame]["decay_time_traj_score"].to_numpy()
        == instances_pred[test_frame]["final_traj_score"].to_numpy()
    ).all()
    assert asso_equals

    assert len(instances_pred[test_frame]["pred_track_ids"] == num_detected)


#@pytest.mark.parametrize("set_default_device", ["cpu"], indirect=True)
def test_post_processing(): #set_default_device
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
