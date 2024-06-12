"""Module containing Boxes class."""

import torch
from typing import Self


class Boxes:
    """Adapted from https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/boxes.py.

    This structure stores a list of boxes as a Nx4 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx4. Each row is (x1, y1, x2, y2).
    """

    def __init__(self, tensor: torch.Tensor):
        """Initialize Boxes.

        Args:
            tensor (Tensor[float]): a Nx4 matrix.  Each row is (x1, y1, x2, y2).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(
                tensor, dtype=torch.float32, device=torch.device("cpu")
            )
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 3 and tensor.size(-1) == 4, tensor.size()

        self.tensor = tensor

    def clone(self) -> Self:
        """Clone the Boxes.

        Returns:
            Boxes
        """
        return Boxes(self.tensor.clone())

    def to(self, device: torch.device) -> Self:
        """Load boxes to gpu/cpu.

        Args:
            device: The device to load the boxes to

        Returns: Boxes on device.
        """
        # Boxes are assumed float32 and does not support to(dtype)
        return Boxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """Compute the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor
        area = (box[:, :, 2] - box[:, :, 0]) * (box[:, :, 3] - box[:, :, 1])
        return area

    def clip(self, box_size: list[int, int]) -> None:
        """Clip (in place) the boxes.

        Limits x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, :, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, :, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, :, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, :, 3].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """Find boxes that are non-empty.

        A box is considered empty, if either of its side is no larger than threshold.

        Args:
            threshold: the smallest a box can be.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor
        widths = box[:, :, 2] - box[:, :, 0]
        heights = box[:, :, 3] - box[:, :, 1]
        keep = (widths > threshold) & (heights > threshold)
        return keep

    def __getitem__(self, item: int | slice | torch.BoolTensor) -> "Boxes":
        """Getter for boxes.

        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        Usage:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
            2. `new_boxes = boxes[2:10]`: return a slice of boxes.
            3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
            with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        NOTE: that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return Boxes(self.tensor[item])
        b = self.tensor[item]
        assert (
            b.dim() == 3
        ), "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return Boxes(b)

    def __len__(self) -> int:
        """Get the number of boxes stored in this object.

        Returns:
            the number of boxes stored in this object
        """
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        """Override representation for printing.

        Returns:
            'Boxes(tensor)'
        """
        return "Boxes(" + str(self.tensor) + ")"

    def inside_box(
        self, box_size: tuple[int, int], boundary_threshold: int = 0
    ) -> torch.Tensor:
        """Check if box is inside reference box.

        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """Get the centroid of the bbox.

        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return (self.tensor[:, :, :2] + self.tensor[:, :, 2:]) / 2

    def scale(self, scale_x: float, scale_y: float) -> None:
        """Scale the box with horizontal and vertical scaling factors."""
        self.tensor[:, :, 0::2] *= scale_x
        self.tensor[:, :, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: list["Boxes"]) -> "Boxes":
        """Concatenates a list of Boxes into a single Boxes.

        Arguments:
            boxes_list: list of `Boxes`

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, Boxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self) -> torch.device:
        """Get the device the box is on.

        Returns: the device the box is on
        """
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """Yield a box as a Tensor of shape (4,) at a time."""
        yield from self.tensor
