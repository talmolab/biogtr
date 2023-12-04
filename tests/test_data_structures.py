"""Tests for Instance, Frame, and TrackQueue Object"""
from biogtr.data_structures import Instance, Frame
from biogtr.inference.track_queue import TrackQueue
import torch


def test_instance():
    """Test Instance object logic."""

    gt_track_id = 0
    pred_track_id = 0
    bbox = torch.randn((1, 4))
    crop = torch.randn((1, 3, 128, 128))
    features = torch.randn((1, 64))

    instance = Instance(
        gt_track_id=gt_track_id,
        pred_track_id=pred_track_id,
        bbox=bbox,
        crop=crop,
        features=features,
    )

    assert instance.has_gt_track_id()
    assert instance.gt_track_id.item() == gt_track_id
    assert instance.has_pred_track_id()
    assert instance.pred_track_id.item() == pred_track_id
    assert instance.has_bbox()
    assert torch.equal(instance.bbox, bbox)
    assert instance.has_features()
    assert torch.equal(instance.features, features)

    instance.gt_track_id = 1
    instance.pred_track_id = 1
    instance.bbox = torch.randn((1, 4))
    instance.crop = torch.randn((1, 3, 128, 128))
    instance.features = torch.randn((1, 64))

    assert instance.has_gt_track_id()
    assert instance.gt_track_id.item() != gt_track_id
    assert instance.has_pred_track_id()
    assert instance.pred_track_id.item() != pred_track_id
    assert instance.has_bbox()
    assert not torch.equal(instance.bbox, bbox)
    assert instance.has_features()
    assert not torch.equal(instance.features, features)

    instance.gt_track_id = None
    instance.pred_track_id = -1
    instance.bbox = None
    instance.crop = None
    instance.features = None

    assert not instance.has_gt_track_id()
    assert instance.gt_track_id.shape[0] == 0
    assert not instance.has_pred_track_id()
    assert instance.pred_track_id.item() != pred_track_id
    assert not instance.has_bbox()
    assert not torch.equal(instance.bbox, bbox)
    assert not instance.has_features()
    assert not torch.equal(instance.features, features)


def test_frame():
    n_detected = 2
    n_traj = 3
    video_id = 0
    frame_id = 0
    img_shape = torch.tensor([3, 1024, 1024])
    asso_output = torch.randn(n_detected, 16)
    traj_score = torch.randn(n_detected, n_traj)
    matches = ([0, 1], [0, 1])

    instances = []
    for i in range(n_detected):
        instances.append(
            Instance(
                gt_track_id=i,
                pred_track_id=i,
                bbox=torch.randn(1, 4),
                crop=torch.randn(1, 3, 64, 64),
                features=torch.randn(1, 64),
            )
        )
    frame = Frame(
        video_id=video_id, frame_id=frame_id, img_shape=img_shape, instances=instances
    )

    assert frame.video_id.item() == video_id
    assert frame.frame_id.item() == frame_id
    assert torch.equal(frame.img_shape, img_shape)
    assert frame.num_detected == n_detected
    assert frame.has_instances()
    assert len(frame.instances) == n_detected
    assert frame.has_gt_track_ids()
    assert len(frame.get_gt_track_ids()) == n_detected
    assert frame.has_pred_track_ids()
    assert len(frame.get_pred_track_ids()) == n_detected
    assert not frame.has_matches()
    assert not frame.has_asso_output()
    assert not frame.has_traj_score()

    frame.asso_output = asso_output
    frame.add_traj_score("initial", traj_score)
    frame.matches = matches

    assert frame.has_matches()
    assert frame.matches == matches
    assert frame.has_asso_output()
    assert torch.equal(frame.asso_output, asso_output)
    assert frame.has_traj_score()
    assert torch.equal(frame.get_traj_score("initial"), traj_score)

    frame.instances = []

    assert frame.video_id.item() == video_id
    assert frame.num_detected == 0
    assert not frame.has_instances()
    assert len(frame.instances) == 0
    assert not frame.has_gt_track_ids()
    assert not len(frame.get_gt_track_ids())
    assert not frame.has_pred_track_ids()
    assert len(frame.get_pred_track_ids()) == 0
    assert frame.has_matches()
    assert frame.has_asso_output()
    assert frame.has_traj_score()


def test_track_queue():
    window_size = 8
    max_gap = 10
    img_shape = (3, 1024, 1024)
    n_instances_per_frame = [2] * window_size

    frames = []
    instances_per_frame = []

    tq = TrackQueue(window_size, max_gap)
    for i in range(window_size):
        instances = []
        for j in range(n_instances_per_frame[i]):
            instances.append(Instance(gt_track_id=j, pred_track_id=j))
        instances_per_frame.append(instances)
        frame = Frame(video_id=0, frame_id=i, img_shape=img_shape, instances=instances)
        frames.append(frame)

        tq.add_frame(frame)

    assert len(tq) == sum(n_instances_per_frame[1:])
    assert tq.n_tracks == max(n_instances_per_frame)
    assert tq.tracks == [i for i in range(max(n_instances_per_frame))]
    assert len(tq.collate_tracks()) == window_size - 1
    assert all([gap == 0 for gap in tq._curr_gap.values()])
    assert tq.curr_track == max(n_instances_per_frame) - 1

    tq.add_frame(
        Frame(
            video_id=0,
            frame_id=window_size + 1,
            img_shape=img_shape,
            instances=[Instance(gt_track_id=0, pred_track_id=0)],
        )
    )

    assert len(tq._queues[0]) == window_size - 1
    assert tq._curr_gap[0] == 0
    assert tq._curr_gap[max(n_instances_per_frame) - 1] == 1

    tq.add_frame(
        Frame(
            video_id=0,
            frame_id=window_size + 1,
            img_shape=img_shape,
            instances=[
                Instance(gt_track_id=1, pred_track_id=1),
                Instance(
                    gt_track_id=max(n_instances_per_frame),
                    pred_track_id=max(n_instances_per_frame),
                ),
            ],
        )
    )

    assert len(tq._queues[max(n_instances_per_frame)]) == 1
    assert tq._curr_gap[1] == 0
    assert tq._curr_gap[0] == 1

    for i in range(max_gap):
        tq.add_frame(
            Frame(
                video_id=0,
                frame_id=window_size + i + 1,
                img_shape=img_shape,
                instances=[Instance(gt_track_id=0, pred_track_id=0)],
            )
        )

    assert tq.n_tracks == 1
    assert tq.curr_track == max(n_instances_per_frame)
    assert 0 in tq._queues.keys()

    tq.end_tracks()

    assert len(tq) == 0
