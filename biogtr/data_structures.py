"""Module containing data classes such as Instances and Frames."""
import torch
from numpy.typing import ArrayLike
from typing import Union, List


class Instance:
    """Class representing a single instance to be tracked."""

    def __init__(
        self,
        gt_track_id: int = None,
        pred_track_id: int = -1,
        bbox: ArrayLike = torch.empty((0, 4)),
        crop: ArrayLike = torch.tensor([]),
        features: ArrayLike = torch.tensor([]),
        device: str = None,
    ):
        """Initialize Instance.

        Args:
            gt_track_id: Ground truth track id - only used for train/eval.
            pred_track_id: Predicted track id. Untracked instance is represented by -1.
            bbox: The bounding box coordinate of the instance. Defaults to an empty tensor.
            crop: The crop of the instance.
            features: The reid features extracted from the CNN backbone used in the transformer.
            device: String representation of the device the instance should be on.
        """
        if gt_track_id is not None:
            self._gt_track_id = torch.tensor([gt_track_id])
        else:
            self._gt_track_id = torch.tensor([])

        if pred_track_id is not None:
            self._pred_track_id = torch.tensor([pred_track_id])
        else:
            self._pred_track_id = torch.tensor([])

        if not isinstance(bbox, torch.Tensor):
            self._bbox = torch.tensor(bbox)
        else:
            self._bbox = bbox

        if self._bbox.shape[0] and len(self._bbox.shape) == 1:
            self._bbox = self._bbox.unsqueeze(0)

        if not isinstance(crop, torch.Tensor):
            self._crop = torch.tensor(crop)
        else:
            self._crop = crop

        if len(self._crop.shape) == 2:
            self._crop = self._crop.unsqueeze(0).unsqueeze(0)
        elif len(self._crop.shape) == 3:
            self._crop = self._crop.unsqueeze(0)

        if not isinstance(crop, torch.Tensor):
            self._features = torch.tensor(features)
        else:
            self._features = features

        if self._features.shape[0] and len(self._features.shape) == 1:
            self._features = self._features.unsqueeze(0)

        self._device = device
        self.to(self._device)

    def __repr__(self) -> str:
        """Return string representation of the Instance."""
        return (
            "Instance("
            f"gt_track_id={self._gt_track_id.item()}, "
            f"pred_track_id={self._pred_track_id.item()}, "
            f"bbox={self._bbox}, "
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
        self._gt_track_id = self._gt_track_id.to(map_location)
        self._pred_track_id = self._pred_track_id.to(map_location)
        self._bbox = self._bbox.to(map_location)
        self._crop = self._crop.to(map_location)
        self._features = self._features.to(map_location)
        self.device = map_location
        return self

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

    def has_bbox(self) -> bool:
        """Determine if the instance has a bbox.

        Returns:
            True if the instance has a bounding box, false otherwise.
        """
        if self._bbox.shape[0] == 0:
            return False
        else:
            return True

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
            self._crop = self._crop.unsqueeze(0).unsqueeze(0)
        elif len(self._crop.shape) == 3:
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


class Frame:
    """Data structure containing metadata for a single frame of a video."""

    def __init__(
        self,
        video_id: int,
        frame_id: int,
        img_shape: ArrayLike = [0, 0, 0],
        instances: List[Instance] = [],
        asso_output: ArrayLike = None,
        matches: tuple = None,
        traj_score: Union[ArrayLike, dict] = None,
        device=None,
    ):
        """Initialize Frame.

        Args:
            video_id: The video index in the dataset.
            frame_id: The index of the frame in a video.
            img_shape: The shape of the original frame (not the crop).
            instances: A list of Instance objects that appear in the frame.
            asso_output: The association matrix between instances
            output directly from the transformer.
            matches: matches from LSA algorithm between the instances and
            available trajectories during tracking.
            traj_score: Either a dict containing the association matrix
            between instances and trajectories along postprocessing pipeline
            or a single association matrix.
            device: The device the frame should be moved to.
        """
        self._video_id = torch.tensor([video_id])
        self._frame_id = torch.tensor([frame_id])

        if isinstance(img_shape, torch.Tensor):
            self._img_shape = img_shape
        else:
            self._img_shape = torch.tensor([img_shape])

        self._instances = instances

        self._asso_output = asso_output
        self._matches = matches

        if traj_score is None:
            self._traj_score = {}
        elif isinstance(traj_score, dict):
            self._traj_score = traj_score
        else:
            self._traj_score = {"initial": traj_score}

        self._device = device
        self.to(device)

    def __repr__(self) -> str:
        """Return String representation of the Frame.

        Returns:
            The string representation of the frame.
        """
        return (
            "Frame("
            f"video_id={self._video_id.item()}, "
            f"frame_id={self._frame_id.item()}, "
            f"img_shape={self._img_shape}, "
            f"num_detected={self.num_detected}, "
            f"asso_output={self._asso_output}, "
            f"traj_score={self._traj_score}, "
            f"matches={self._matches}, "
            f"instances={self._instances}, "
            f"device={self._device}"
            ")"
        )

    def to(self, map_location: str):
        """Move frame to different device or dtype (See `torch.to` for more info).

        Args:
            map_location: A string representing the device to move to.

        Returns:
            The frame moved to a different device/dtype.
        """
        self._video_id = self._video_id.to(map_location)
        self._frame_id = self._frame_id.to(map_location)
        self._img_shape = self._img_shape.to(map_location)

        if isinstance(self._asso_output, torch.Tensor):
            self._asso_output = self._asso_output.to(map_location)

        if isinstance(self._matches, torch.Tensor):
            self._matches = self._matches.to(map_location)

        for key, val in self._traj_score.items():
            if isinstance(val, torch.Tensor):
                self._traj_score[key] = val.to(map_location)

        for instance in self._instances:
            instance = instance.to(map_location)

        self._device = map_location
        return self

    @property
    def device(self) -> str:
        """The device the frame is on.

        Returns:
            The string representation of the device the frame is on.
        """
        return self._device

    @device.setter
    def device(self, device: str) -> None:
        """Set the device.

        Note: Do not set `frame.device = device` normally. Use `frame.to(device)` instead.

        Args:
            device: the device the function should be on.
        """
        self._device = device

    @property
    def video_id(self) -> torch.Tensor:
        """The index of the video the frame comes from.

        Returns:
            A tensor containing the video index.
        """
        return self._video_id

    @video_id.setter
    def video_id(self, video_id: int) -> None:
        """Set the video index.

        Note: Generally the video_id should be immutable after initialization.

        Args:
            video_id: an int representing the index of the video that the frame came from.
        """
        self._video_id = torch.tensor([video_id])

    @property
    def frame_id(self) -> torch.Tensor:
        """The index of the frame in a full video.

        Returns:
            A torch tensor containing the index of the frame in the video.
        """
        return self._frame_id

    @frame_id.setter
    def frame_id(self, frame_id: int) -> None:
        """Set the frame index of the frame.

        Note: The frame_id should generally be immutable after initialization.

        Args:
            frame_id: The int index of the frame in the full video.
        """
        self._frame_id = torch.tensor([frame_id])

    @property
    def img_shape(self) -> torch.Tensor:
        """The shape of the pre-cropped frame.

        Returns:
            A torch tensor containing the shape of the frame. Should generally be (c, h, w)
        """
        return self._img_shape

    @img_shape.setter
    def img_shape(self, img_shape: ArrayLike) -> None:
        """Set the shape of the frame image.

        Note: the img_shape should generally be immutable after initialization.

        Args:
            img_shape: an ArrayLike object containing the shape of the frame image.
        """
        if isinstance(img_shape, torch.Tensor):
            self._img_shape = img_shape
        else:
            self._img_shape = torch.tensor([img_shape])

    @property
    def instances(self) -> List[Instance]:
        """A list of instances in the frame.

        Returns:
            The list of instances that appear in the frame.
        """
        return self._instances

    @instances.setter
    def instances(self, instances: List[Instance]) -> None:
        """Set the frame's instance.

        Args:
            instances: A list of Instances that appear in the frame.
        """
        self._instances = instances

    def has_instances(self) -> bool:
        """Determine whether there are instances in the frame.

        Returns:
            True if there are instances in the frame, otherwise False.
        """
        if self.num_detected == 0:
            return False
        return True

    @property
    def num_detected(self) -> int:
        """The number of instances in the frame.

        Returns:
            the number of instances in the frame.
        """
        return len(self.instances)

    @property
    def asso_output(self) -> ArrayLike:
        """The association matrix between instances outputed directly by transformer.

        Returns:
            An arraylike (n_query, n_nonquery) association matrix between instances.
        """
        return self._asso_output

    def has_asso_output(self) -> bool:
        """Determine whether the frame has an association matrix computed.

        Returns:
            True if the frame has an association matrix otherwise, False.
        """
        if self._asso_output is None or len(self._asso_output) == 0:
            return False
        return True

    @asso_output.setter
    def asso_output(self, asso_output: ArrayLike) -> None:
        """Set the association matrix of a frame.

        Args:
            asso_output: An arraylike (n_query, n_nonquery) association matrix between instances.
        """
        self._asso_output = asso_output

    @property
    def matches(self) -> tuple:
        """Matches between frame instances and availabel trajectories.

        Returns:
            A tuple containing the instance idx and trajectory idx for the matched instance.
        """
        return self._matches

    @matches.setter
    def matches(self, matches: tuple) -> None:
        """Set the frame matches.

        Args:
            matches: A tuple containing the instance idx and trajectory idx for the matched instance.
        """
        self._matches = matches

    def has_matches(self) -> bool:
        """Check whether or not matches have been computed for frame.

        Returns:
            True if frame contains matches otherwise False.
        """
        if self._matches is not None and len(self._matches) > 0:
            return True
        return False

    def get_traj_score(self, key=None) -> Union[dict, ArrayLike, None]:
        """Get dictionary containing association matrix between instances and trajectories along postprocessing pipeline.

        Args:
            key: The key of the trajectory score to be accessed.
            Can be one of {None, 'initial', 'decay_time', 'max_center_dist', 'iou', 'final'}

        Returns:
            - dictionary containing all trajectory scores if key is None
            - trajectory score associated with key
            - None if the key is not found
        """
        if key is None:
            return self._traj_score
        else:
            try:
                return self._traj_score[key]
            except KeyError as e:
                print(f"Could not access {key} traj_score due to {e}")
                return None

    def add_traj_score(self, key, traj_score: ArrayLike) -> None:
        """Add trajectory score to dictionary.

        Args:
            key: key associated with traj score to be used in dictionary
            traj_score: association matrix between instances and trajectories
        """
        self._traj_score[key] = traj_score

    def has_traj_score(self) -> bool:
        """Check if any trajectory association matrix has been saved.

        Returns:
            True there is at least one association matrix otherwise, false.
        """
        if len(self._traj_score) == 0:
            return False
        return True

    def has_gt_track_ids(self) -> bool:
        """Check if any of frames instances has a gt track id.

        Returns:
            True if at least 1 instance has a gt track id otherwise False.
        """
        if self.has_instances():
            return any([instance.has_gt_track_id() for instance in self.instances])
        return False

    def get_gt_track_ids(self) -> torch.Tensor:
        """Get the gt track ids of all instances in the frame.

        Returns:
            an (N,) shaped tensor with the gt track ids of each instance in the frame.
        """
        if not self.has_instances():
            return torch.tensor([])
        return torch.cat([instance.gt_track_id for instance in self.instances])

    def has_pred_track_ids(self) -> bool:
        """Check if any of frames instances has a pred track id.

        Returns:
            True if at least 1 instance has a pred track id otherwise False.
        """
        if self.has_instances():
            return any([instance.has_pred_track_id() for instance in self.instances])
        return False

    def get_pred_track_ids(self) -> torch.Tensor:
        """Get the pred track ids of all instances in the frame.

        Returns:
            an (N,) shaped tensor with the pred track ids of each instance in the frame.
        """
        if not self.has_instances():
            return torch.tensor([])
        return torch.cat([instance.pred_track_id for instance in self.instances])

    def has_bboxes(self) -> bool:
        """Check if any of frames instances has a bounding box.

        Returns:
            True if at least 1 instance has a bounding box otherwise False.
        """
        if self.has_instances():
            return any([instance.has_bboxes() for instance in self.instances])
        return False

    def get_bboxes(self) -> torch.Tensor:
        """Get the bounding boxes of all instances in the frame.

        Returns:
            an (N,4) shaped tensor with bounding boxes of each instance in the frame.
        """
        if not self.has_instances():
            return torch.empty(0, 4)
        return torch.cat([instance.bbox for instance in self.instances], dim=0)

    def has_crops(self) -> bool:
        """Check if any of frames instances has a crop.

        Returns:
            True if at least 1 instance has a crop otherwise False.
        """
        if self.has_instances():
            return any([instance.has_crop() for instance in self.instances])
        return False

    def get_crops(self) -> torch.Tensor:
        """Get the crops of all instances in the frame.

        Returns:
            an (N, C, H, W) shaped tensor with crops of each instance in the frame.
        """
        if not self.has_instances():
            return torch.tensor([])
        return torch.cat([instance.crop for instance in self.instances], dim=0)

    def has_features(self):
        """Check if any of frames instances has reid features already computed.

        Returns:
            True if at least 1 instance have reid features otherwise False.
        """
        if self.has_instances():
            return any([instance.has_features() for instance in self.instances])
        return False

    def get_features(self):
        """Get the reid feature vectors of all instances in the frame.

        Returns:
            an (N, D) shaped tensor with reid feature vectors of each instance in the frame.
        """
        if not self.has_instances():
            return torch.tensor([])
        return torch.cat([instance.features for instance in self.instances], dim=0)
