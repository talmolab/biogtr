"""Module containing data structures for storing instances of the same Track."""

from __future__ import annotations
import attrs


@attrs.define(eq=False)
class Track:
    """Object for storing instances of the same track.

    Attributes:
        id: the track label.
        instances: A list of instances belonging to the track.
    """

    _id: int = attrs.field(alias="id")
    _instances: list["Instance"] = attrs.field(alias="instances", factory=list)

    def __repr__(self) -> str:
        """Get the string representation of the track.

        Returns:
            the string representation of the Track.
        """
        return f"Track(id={self.id}, len={len(self)})"

    @property
    def track_id(self) -> int:
        """Get the id of the track.

        Returns:
            The integer id of the track.
        """
        return self._id

    @track_id.setter
    def track_id(self, track_id: int) -> None:
        """Set the id of the track.

        Args:
            track_id: the int id of the track.
        """
        self._id = track_id

    @property
    def instances(self) -> list["Instances"]:
        """Get the instances belonging to this track.

        Returns:
            A list of instances with this track id.
        """
        return self._instances

    @instances.setter
    def instances(self, instances) -> None:
        """Set the instances belonging to this track.

        Args:
            instances: A list of instances that belong to the same track.
        """
        self._instances = instances

    @property
    def frames(self) -> set["Frame"]:
        """Get the frames where this track appears.

        Returns:
            A set of `Frame` objects where this track appears.
        """
        return set([instance.frame for instance in self.instances])

    def __len__(self) -> int:
        """Get the length of the track.

        Returns:
            The number of instances/frames in the track.
        """
        return len(self.instances)

    def __getitem__(self, ind: int | list[int]) -> "Instance" | list["Instance"]:
        """Get an instance from the track.

        Args:
            ind: Either a single int or list of int indices.

        Returns:
            the instance at that index of the track.instances.
        """
        if isinstance(ind, int):
            return self.instances[ind]
        elif isinstance(ind, list):
            return [self.instances[i] for i in ind]
        else:
            raise ValueError(f"Ind must be an int or list of ints, found {type(ind)}")
