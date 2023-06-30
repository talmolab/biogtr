"""Test dataset logic."""
from biogtr.datasets.base_dataset import BaseDataset
from biogtr.datasets.data_utils import get_max_padding, FixedRandomSampler
from biogtr.datasets.microscopy_dataset import MicroscopyDataset
from biogtr.datasets.sleap_dataset import SleapDataset
from biogtr.datasets.tracking_dataset import TrackingDataset
from biogtr.datasets.cell_tracking_dataset import CellTrackingDataset
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pytest
import torch


def test_base_dataset():
    """Test BaseDataset logic."""

    class DummyDataset(BaseDataset):
        pass

    ds = DummyDataset(
        files=[], padding=0, crop_size=0, chunk=False, clip_length=0, mode=""
    )

    with pytest.raises(NotImplementedError):
        ds.get_indices(0)

    with pytest.raises(NotImplementedError):
        ds.get_instances([], [])

    with pytest.raises(NotImplementedError):
        ds.__getitem__(0)

    with pytest.raises(AttributeError):
        ds.__len__()


def test_sleap_dataset(two_flies):
    """Test sleap dataset logic.

    Args:
        two_flies: two flies fixture used for testing
    """
    clip_length = 32

    train_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
    )

    instances = next(iter(train_ds))

    assert len(instances) == clip_length
    assert len(instances[0]["gt_track_ids"]) == 2
    assert len(instances[0]["gt_track_ids"]) == instances[0]["num_detected"].item()


def test_icy_dataset(ten_icy_particles):
    """Test icy dataset logic.

    Args:
        ten_icy_particles: icy fixture used for testing
    """
    clip_length = 8

    train_ds = MicroscopyDataset(
        videos=[ten_icy_particles[0]],
        tracks=[ten_icy_particles[1]],
        source="icy",
        crop_size=64,
        chunk=True,
        clip_length=clip_length,
    )

    instances = next(iter(train_ds))

    assert len(instances) == clip_length
    assert len(instances[0]["gt_track_ids"]) == 10
    assert len(instances[0]["gt_track_ids"]) == instances[0]["num_detected"].item()


def test_trackmate_dataset(trackmate_lysosomes):
    """Test trackmate dataset logic.

    Args:
        trackmate_lysosomes: trackmate fixture used for testing
    """
    clip_length = 8

    train_ds = MicroscopyDataset(
        videos=[trackmate_lysosomes[0]],
        tracks=[trackmate_lysosomes[1]],
        source="trackmate",
        crop_size=64,
        chunk=True,
        clip_length=clip_length,
    )

    instances = next(iter(train_ds))

    assert len(instances) == clip_length
    assert len(instances[0]["gt_track_ids"]) == 26
    assert len(instances[0]["gt_track_ids"]) == instances[0]["num_detected"].item()


def test_isbi_dataset(isbi_microtubules, isbi_receptors):
    """Test isbi dataset logic.

    Args:
        isbi_microtubules: isbi microtubules fixture used for testing
        isbi_receptors: isbi receptors fixture used for testing
    """

    clip_length = 8

    for ds in [isbi_microtubules, isbi_receptors]:
        num_objects = 47 if ds == isbi_microtubules else 61

        train_ds = MicroscopyDataset(
            videos=[ds[0]],
            tracks=[ds[1]],
            source="isbi",
            crop_size=64,
            chunk=True,
            clip_length=clip_length,
        )

        instances = next(iter(train_ds))

        assert len(instances) == clip_length
        assert len(instances[0]["gt_track_ids"]) == num_objects
        assert len(instances[0]["gt_track_ids"]) == instances[0]["num_detected"].item()


def test_cell_tracking_dataset(cell_tracking):
    """Test cell tracking dataset logic.

    Args:
        cell_tracking: HL60 nuclei fixture used for testing
    """

    clip_length = 8

    train_ds = CellTrackingDataset(
        raw_images=[cell_tracking[0]],
        gt_images=[cell_tracking[1]],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
        gt_list=cell_tracking[2],
    )

    instances = next(iter(train_ds))

    gt_track_ids_1 = instances[0]["gt_track_ids"]

    assert len(instances) == clip_length
    assert len(gt_track_ids_1) == 30
    assert len(gt_track_ids_1) == instances[0]["num_detected"].item()

    # fall back to using np.unique when gt_list not available
    train_ds = CellTrackingDataset(
        raw_images=[cell_tracking[0]],
        gt_images=[cell_tracking[1]],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
    )

    instances = next(iter(train_ds))

    gt_track_ids_2 = instances[0]["gt_track_ids"]

    assert len(instances) == clip_length
    assert len(gt_track_ids_2) == 30
    assert len(gt_track_ids_2) == instances[0]["num_detected"].item()
    assert gt_track_ids_1.all() == gt_track_ids_2.all()


def test_tracking_dataset(two_flies):
    """Test lightning dataset logic.

    Args:
        two_flies: two flies fixture used for testing
    """
    batch_size = 2
    clip_length = 16
    num_workers = 0
    pin_memory = num_workers > 0
    generator = torch.Generator(device="cuda") if torch.cuda.is_available() else None

    train_sleap_ds = SleapDataset(
        [two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=True,
        clip_length=clip_length,
    )

    train_sleap_dl = DataLoader(
        dataset=train_sleap_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=train_sleap_ds.no_batching_fn,
        pin_memory=pin_memory,
        generator=generator,
    )

    val_sleap_ds = SleapDataset(
        [two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=False,
        clip_length=clip_length,
    )

    val_sleap_dl = DataLoader(
        dataset=val_sleap_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_sleap_ds.no_batching_fn,
        pin_memory=pin_memory,
        generator=generator,
    )

    test_sleap_ds = SleapDataset(
        [two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=False,
        clip_length=clip_length,
    )

    test_sleap_dl = DataLoader(
        dataset=test_sleap_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_sleap_ds.no_batching_fn,
        pin_memory=pin_memory,
        generator=generator,
    )

    # test default if dls are None
    tracking_ds = TrackingDataset(
        train_ds=train_sleap_ds, val_ds=val_sleap_ds, test_ds=test_sleap_ds
    )

    assert (
        tracking_ds.train_dataloader().batch_size == 1
        and tracking_ds.val_dataloader().batch_size == 1
        and tracking_ds.test_dataloader().batch_size == 1
    )

    # test using all dls
    tracking_ds = TrackingDataset(
        train_dl=train_sleap_dl, val_dl=val_sleap_dl, test_dl=test_sleap_dl
    )

    assert (
        tracking_ds.train_dataloader().batch_size == 2
        and tracking_ds.val_dataloader().batch_size == 2
        and tracking_ds.test_dataloader().batch_size == 2
    )

    # test using overrided dls over defaults
    tracking_ds = TrackingDataset(
        train_ds=train_sleap_ds,
        train_dl=train_sleap_dl,
        val_ds=val_sleap_ds,
        val_dl=val_sleap_dl,
        test_ds=test_sleap_ds,
        test_dl=test_sleap_dl,
    )

    assert (
        tracking_ds.train_dataloader().batch_size == 2
        and tracking_ds.val_dataloader().batch_size == 2
        and tracking_ds.test_dataloader().batch_size == 2
    )
    # test using overrided dls over defaults with combinations
    tracking_ds = TrackingDataset(
        train_ds=train_sleap_ds,
        val_ds=val_sleap_ds,
        val_dl=val_sleap_dl,
        test_ds=test_sleap_ds,
        test_dl=test_sleap_dl,
    )

    assert (
        tracking_ds.train_dataloader().batch_size == 1
        and tracking_ds.val_dataloader().batch_size == 2
        and tracking_ds.test_dataloader().batch_size == 2
    )


def test_augmentations(two_flies, ten_icy_particles):
    """Test augmentations.

    Args:
        two_flies: flies fixture used for testing
        ten_icy_particles: icy fixture used for testing
    """

    no_augs_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=True,
        clip_length=8,
    )

    no_augs_instances = next(iter(no_augs_ds))

    augmentations = {
        "Rotate": {"limit": 45, "p": 1.0},
        "GaussianBlur": {"blur_limit": (3, 7), "sigma_limit": 0, "p": 1.0},
        "GaussNoise": {
            "var_limit": (10.0, 50.0),
            "mean": 0,
            "per_channel": True,
            "p": 1.0,
        },
    }

    augs_ds = SleapDataset(
        slp_files=[two_flies[0]],
        video_files=[two_flies[1]],
        crop_size=128,
        chunk=True,
        clip_length=8,
        augmentations=augmentations,
    )

    augs_instances = next(iter(augs_ds))

    a = no_augs_instances[0]["crops"]
    b = augs_instances[0]["crops"]

    assert not torch.all(a.eq(b))

    no_augs_ds = MicroscopyDataset(
        videos=[ten_icy_particles[0]],
        tracks=[ten_icy_particles[1]],
        source="icy",
        crop_size=64,
        chunk=True,
        clip_length=8,
    )

    no_augs_instances = next(iter(no_augs_ds))

    # pad before rotation
    padded_height, padded_width = get_max_padding(512, 512)

    pad = {
        "PadIfNeeded": {
            "min_height": padded_height,
            "min_width": padded_width,
            "border_mode": 0,
        }
    }

    pad.update(augmentations)
    augmentations = pad

    # add another to end
    motion = {"MotionBlur": {"blur_limit": (3, 7), "p": 0.5}}

    augmentations.update(motion)

    augs_ds = MicroscopyDataset(
        videos=[ten_icy_particles[0]],
        tracks=[ten_icy_particles[1]],
        source="icy",
        crop_size=64,
        chunk=True,
        clip_length=8,
        augmentations=augmentations,
    )

    augs_instances = next(iter(augs_ds))

    a = no_augs_instances[0]["crops"]
    b = augs_instances[0]["crops"]

    assert not torch.all(a.eq(b))


def test_fixed_random_sampler():
    """Test FixedRandomSampler logic."""

    # dummy dataset
    dataset = TensorDataset(torch.rand(100, 10))

    random_seed = 12345

    # Test only seed
    sampler = FixedRandomSampler(dataset, seed=random_seed)
    sample_indices = list(iter(sampler))
    assert len(sample_indices) == len(dataset)
    assert len(set(sample_indices)) <= len(dataset)

    # Test both seed and num_epochs
    sampler = FixedRandomSampler(dataset, seed=random_seed, num_epochs=2)
    sample_indices = list(iter(sampler))
    assert len(sample_indices) == 2 * len(dataset)

    # Test fixed seed sampling consistency
    sampler1 = FixedRandomSampler(dataset, seed=random_seed)
    sampler2 = FixedRandomSampler(dataset, seed=random_seed)
    sample_indices1 = list(iter(sampler1))
    sample_indices2 = list(iter(sampler2))
    np.testing.assert_array_equal(sample_indices1, sample_indices2)

    # Test different seeds give different results
    sampler1 = FixedRandomSampler(dataset, seed=random_seed)
    sampler2 = FixedRandomSampler(dataset, seed=random_seed + 1)
    sample_indices1 = list(iter(sampler1))
    sample_indices2 = list(iter(sampler2))
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(sample_indices1, sample_indices2)
