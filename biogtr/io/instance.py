"""Module containing data class for storing detections"""

import torch
import sleap_io as sio
import numpy as np
from numpy.typing import ArrayLike
from typing import Union, List


class Instance:
    """Class representing a single instance to be tracked."""

    def __init__(
        self,
        gt_track_id: int = -1,
        pred_track_id: int = -1,
        bbox: ArrayLike = None,
        crop: ArrayLike = None,
        centroid: dict[str, ArrayLike] = None,
        features: ArrayLike = None,
        track_score: float = -1.0,
        point_scores: ArrayLike = None,
        instance_score: float = -1.0,
        skeleton: sio.Skeleton = None,
        pose: dict[str, ArrayLike] = None,
        device: str = None,
    ):
        """Initialize Instance.

        Args:
            gt_track_id: Ground truth track id - only used for train/eval.
            pred_track_id: Predicted track id. Untracked instance is represented by -1.
            bbox: The bounding box coordinate of the instance. Defaults to an empty tensor.
            crop: The crop of the instance.
            centroid: the centroid around which the bbox was cropped.
            features: The reid features extracted from the CNN backbone used in the transformer.
            track_score: The track score output from the association matrix.
            point_scores: The point scores from sleap.
            instance_score: The instance scores from sleap.
            skeleton: The sleap skeleton used for the instance.
            pose: A dictionary containing the node name and corresponding point.
            device: String representation of the device the instance should be on.
        """
        if gt_track_id is not None:
            self._gt_track_id = torch.tensor([gt_track_id])
        else:
            self._gt_track_id = torch.tensor([-1])

        if pred_track_id is not None:
            self._pred_track_id = torch.tensor([pred_track_id])
        else:
            self._pred_track_id = torch.tensor([])

        if skeleton is None:
            self._skeleton = sio.Skeleton(["centroid"])
        else:
            self._skeleton = skeleton

        if bbox is None:
            self._bbox = torch.empty(1, 0, 4)

        elif not isinstance(bbox, torch.Tensor):
            self._bbox = torch.tensor(bbox)

        else:
            self._bbox = bbox

        if self._bbox.shape[0] and len(self._bbox.shape) == 1:
            self._bbox = self._bbox.unsqueeze(0)  # (n_anchors, 4)

        if self._bbox.shape[1] and len(self._bbox.shape) == 2:
            self._bbox = self._bbox.unsqueeze(0)  # (1, n_anchors, 4)

        if centroid is not None:
            self._centroid = centroid

        elif self.bbox.shape[1]:
            y1, x1, y2, x2 = self.bbox.squeeze(dim=0).nanmean(dim=0)
            self._centroid = {"centroid": np.array([(x1 + x2) / 2, (y1 + y2) / 2])}

        else:
            self._centroid = {}

        if crop is None:
            self._crop = torch.tensor([])
        elif not isinstance(crop, torch.Tensor):
            self._crop = torch.tensor(crop)
        else:
            self._crop = crop

        if len(self._crop.shape) == 2:  # (h, w)
            self._crop = self._crop.unsqueeze(0)  # (c, h, w)
        if len(self._crop.shape) == 3:
            self._crop = self._crop.unsqueeze(0)  # (1, c, h, w)

        if features is None:
            self._features = torch.tensor([])
        elif not isinstance(features, torch.Tensor):
            self._features = torch.tensor(features)
        else:
            self._features = features

        if self._features.shape[0] and len(self._features.shape) == 1:  # (d,)
            self._features = self._features.unsqueeze(0)  # (1, d)

        if pose is not None:
            self._pose = pose

        elif self.bbox.shape[1]:
            y1, x1, y2, x2 = self.bbox.squeeze(dim=0).mean(dim=0)
            self._pose = {"centroid": np.array([(x1 + x2) / 2, (y1 + y2) / 2])}

        else:
            self._pose = {}

        self._track_score = track_score
        self._instance_score = instance_score

        if point_scores is not None:
            self._point_scores = point_scores
        else:
            self._point_scores = np.zeros_like(self.pose)

        self._device = device
        self.to(self._device)

    def __repr__(self) -> str:
        """Return string representation of the Instance."""
        return (
            "Instance("
            f"gt_track_id={self._gt_track_id.item()}, "
            f"pred_track_id={self._pred_track_id.item()}, "
            f"bbox={self._bbox}, "
            f"centroid={self._centroid}, "
            f"crop={self._crop.shape}, "
            f"features={self._features.shape}, "
            f"device={self._device}"
            ")"
        )

    def to(self, map_location):
        """Move instance to different device or change dtype. (See `torch.to` for more info).

        Args:
            map_location: Either the device or dtype for the instance to be moved.

        Returns:
            self: reference to the instance moved to correct device/dtype.
        """
        if map_location is not None and map_location != "":
            self._gt_track_id = self._gt_track_id.to(map_location)
            self._pred_track_id = self._pred_track_id.to(map_location)
            self._bbox = self._bbox.to(map_location)
            self._crop = self._crop.to(map_location)
            self._features = self._features.to(map_location)
            self.device = map_location

        return self

    def to_slp(
        self, track_lookup: dict[int, sio.Track] = {}
    ) -> tuple[sio.PredictedInstance, dict[int, sio.Track]]:
        """Convert instance to sleap_io.PredictedInstance object.

        Args:
            track_lookup: A track look up dictionary containing track_id:sio.Track.
        Returns: A sleap_io.PredictedInstance with necessary metadata
        and a track_lookup dictionary to persist tracks.
        """
        try:
            track_id = self.pred_track_id.item()
            if track_id not in track_lookup:
                track_lookup[track_id] = sio.Track(name=self.pred_track_id.item())

            track = track_lookup[track_id]

            return (
                sio.PredictedInstance.from_numpy(
                    points=self.pose,
                    skeleton=self.skeleton,
                    point_scores=self.point_scores,
                    instance_score=self.instance_score,
                    tracking_score=self.track_score,
                    track=track,
                ),
                track_lookup,
            )
        except Exception as e:
            print(
                f"Pose shape: {self.pose.shape}, Pose score shape {self.point_scores.shape}"
            )
            raise RuntimeError(f"Failed to convert to sio.PredictedInstance: {e}")

    @property
    def device(self) -> str:
        """The device the instance is on.

        Returns:
            The str representation of the device the gpu is on.
        """
        return self._device

    @device.setter
    def device(self, device) -> None:
        """Set for the device property.

        Args:
            device: The str representation of the device.
        """
        self._device = device

    @property
    def gt_track_id(self) -> torch.Tensor:
        """The ground truth track id of the instance.

        Returns:
            A tensor containing the ground truth track id
        """
        return self._gt_track_id

    @gt_track_id.setter
    def gt_track_id(self, track: int):
        """Set the instance ground-truth track id.

        Args:
           track: An int representing the ground-truth track id.
        """
        if track is not None:
            self._gt_track_id = torch.tensor([track])
        else:
            self._gt_track_id = torch.tensor([])

    def has_gt_track_id(self) -> bool:
        """Determine if instance has a gt track assignment.

        Returns:
            True if the gt track id is set, otherwise False.
        """
        if self._gt_track_id.shape[0] == 0:
            return False
        else:
            return True

    @property
    def pred_track_id(self) -> torch.Tensor:
        """The track id predicted by the tracker using asso_output from model.

        Returns:
            A tensor containing the predicted track id.
        """
        return self._pred_track_id

    @pred_track_id.setter
    def pred_track_id(self, track: int) -> None:
        """Set predicted track id.

        Args:
            track: an int representing the predicted track id.
        """
        if track is not None:
            self._pred_track_id = torch.tensor([track])
        else:
            self._pred_track_id = torch.tensor([])

    def has_pred_track_id(self) -> bool:
        """Determine whether instance has predicted track id.

        Returns:
            True if instance has a pred track id, False otherwise.
        """
        if self._pred_track_id.item() == -1 or self._pred_track_id.shape[0] == 0:
            return False
        else:
            return True

    @property
    def bbox(self) -> torch.Tensor:
        """The bounding box coordinates of the instance in the original frame.

        Returns:
            A (1,4) tensor containing the bounding box coordinates.
        """
        return self._bbox

    @bbox.setter
    def bbox(self, bbox: ArrayLike) -> None:
        """Set the instance bounding box.

        Args:
            bbox: an arraylike object containing the bounding box coordinates.
        """
        if bbox is None or len(bbox) == 0:
            self._bbox = torch.empty((0, 4))
        else:
            if not isinstance(bbox, torch.Tensor):
                self._bbox = torch.tensor(bbox)
            else:
                self._bbox = bbox

        if self._bbox.shape[0] and len(self._bbox.shape) == 1:
            self._bbox = self._bbox.unsqueeze(0)
        if self._bbox.shape[1] and len(self._bbox.shape) == 2:
            self._bbox = self._bbox.unsqueeze(0)

    def has_bbox(self) -> bool:
        """Determine if the instance has a bbox.

        Returns:
            True if the instance has a bounding box, false otherwise.
        """
        if self._bbox.shape[1] == 0:
            return False
        else:
            return True

    @property
    def centroid(self) -> dict[str, ArrayLike]:
        """The centroid around which the crop was formed.

        Returns:
            A dict containing the anchor name and the x, y bbox midpoint.
        """
        return self._centroid

    @centroid.setter
    def centroid(self, centroid: dict[str, ArrayLike]) -> None:
        """Set the centroid of the instance.

        Args:
            centroid: A dict containing the anchor name and points.
        """
        self._centroid = centroid

    @property
    def anchor(self) -> list[str]:
        """The anchor node name around which the crop was formed.

        Returns:
            the list of anchors around which each crop was formed
            the list of anchors around which each crop was formed
        """
        if self.centroid:
            return list(self.centroid.keys())
        return ""

    @property
    def crop(self) -> torch.Tensor:
        """The crop of the instance.

        Returns:
            A (1, c, h , w) tensor containing the cropped image centered around the instance.
        """
        return self._crop

    @crop.setter
    def crop(self, crop: ArrayLike) -> None:
        """Set the crop of the instance.

        Args:
            crop: an arraylike object containing the cropped image of the centered instance.
        """
        if crop is None or len(crop) == 0:
            self._crop = torch.tensor([])
        else:
            if not isinstance(crop, torch.Tensor):
                self._crop = torch.tensor(crop)
            else:
                self._crop = crop

        if len(self._crop.shape) == 2:
            self._crop = self._crop.unsqueeze(0)
        if len(self._crop.shape) == 3:
            self._crop = self._crop.unsqueeze(0)

    def has_crop(self) -> bool:
        """Determine if the instance has a crop.

        Returns:
            True if the instance has an image otherwise False.
        """
        if self._crop.shape[0] == 0:
            return False
        else:
            return True

    @property
    def features(self) -> torch.Tensor:
        """Re-ID feature vector from backbone model to be used as input to transformer.

        Returns:
            a (1, d) tensor containing the reid feature vector.
        """
        return self._features

    @features.setter
    def features(self, features: ArrayLike) -> None:
        """Set the reid feature vector of the instance.

        Args:
            features: a (1,d) array like object containing the reid features for the instance.
        """
        if features is None or len(features) == 0:
            self._features = torch.tensor([])

        elif not isinstance(features, torch.Tensor):
            self._features = torch.tensor(features)
        else:
            self._features = features

        if self._features.shape[0] and len(self._features.shape) == 1:
            self._features = self._features.unsqueeze(0)

    def has_features(self) -> bool:
        """Determine if the instance has computed reid features.

        Returns:
            True if the instance has reid features, False otherwise.
        """
        if self._features.shape[0] == 0:
            return False
        else:
            return True

    @property
    def pose(self) -> dict[str, ArrayLike]:
        """Get the pose of the instance.

        Returns:
            A dictionary containing the node and corresponding x,y points
        """
        return self._pose

    @pose.setter
    def pose(self, pose: dict[str, ArrayLike]) -> None:
        """Set the pose of the instance.

        Args:
            pose: A nodes x 2 array containing the pose coordinates.
        """
        if pose is not None:
            self._pose = pose

        elif self.bbox.shape[0]:
            y1, x1, y2, x2 = self.bbox.squeeze()
            self._pose = {"centroid": np.array([(x1 + x2) / 2, (y1 + y2) / 2])}

        else:
            self._pose = {}

    def has_pose(self) -> bool:
        """Check if the instance has a pose.

        Returns True if the instance has a pose.
        """
        if len(self.pose):
            return True
        return False

    @property
    def shown_pose(self) -> dict[str, ArrayLike]:
        """Get the pose with shown nodes only.

        Returns: A dictionary filtered by nodes that are shown (points are not nan).
        """
        pose = self.pose
        return {node: point for node, point in pose.items() if not np.isna(point).any()}

    @property
    def skeleton(self) -> sio.Skeleton:
        """Get the skeleton associated with the instance.

        Returns: The sio.Skeleton associated with the instance.
        """
        return self._skeleton

    @skeleton.setter
    def skeleton(self, skeleton: sio.Skeleton) -> None:
        """Set the skeleton associated with the instance.

        Args:
            skeleton: The sio.Skeleton associated with the instance.
        """
        self._skeleton = skeleton

    @property
    def point_scores(self) -> ArrayLike:
        """Get the point scores associated with the pose prediction.

        Returns: a vector of shape n containing the point scores outputed from sleap associated with pose predictions.
        """
        return self._point_scores

    @point_scores.setter
    def point_scores(self, point_scores: ArrayLike) -> None:
        """Set the point scores associated with the pose prediction.

        Args:
            point_scores: a vector of shape n containing the point scores
            outputted from sleap associated with pose predictions.
        """
        self._point_scores = point_scores

    @property
    def instance_score(self) -> float:
        """Get the pose prediction score associated with the instance.

        Returns: a float from 0-1 representing an instance_score.
        """
        return self._instance_score

    @instance_score.setter
    def instance_score(self, instance_score: float) -> None:
        """Set the pose prediction score associated with the instance.

        Args:
            instance_score: a float from 0-1 representing an instance_score.
        """
        self._instance_score = instance_score

    @property
    def track_score(self) -> float:
        """Get the track_score of the instance.

        Returns: A float from 0-1 representing the output used in the tracker for assignment.
        """
        return self._track_score

    @track_score.setter
    def track_score(self, track_score: float) -> None:
        """Set the track_score of the instance.

        Args:
            track_score: A float from 0-1 representing the output used in the tracker for assignment.
        """
        self._track_score = track_score
