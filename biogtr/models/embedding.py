"""Module containing different position and temporal embeddings."""

from typing import Tuple
import math
import torch

# todo: add named tensors, clean variable names


class Embedding(torch.nn.Module):
    """Class that wraps around different embedding types.

    Used for both learned and fixed embeddings.
    """

    def __init__(self):
        """Initialize embeddings."""
        super().__init__()
        # empty init for flexibility
        pass

    def _torch_int_div(
        self, tensor1: torch.Tensor, tensor2: torch.Tensor
    ) -> torch.Tensor:
        """Performs integer division of two tensors.

        Args:
            tensor1: dividend tensor.
            tensor2: divisor tensor.

        Returns:
            torch.Tensor, resulting tensor.
        """
        return torch.div(tensor1, tensor2, rounding_mode="floor")

    def _sine_box_embedding(
        self,
        boxes,
        features: int = 512,
        temperature: int = 10000,
        scale: float = None,
        normalize: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Computes sine positional embeddings for boxes using given parameters.

        Args:
            boxes: the input boxes.
            features: number of position features to use.
            temperature: frequency factor to control spread of pos embed values.
                A higher temp (e.g 10000) gives a larger spread of values
            scale: A scale factor to use if normalizing
            normalize: Whether to normalize the input before computing embedding

        Returns:
            torch.Tensor, the sine positional embeddings.
        """
        # update default parameters with kwargs if available
        params = {
            "features": features,
            "temperature": temperature,
            "scale": scale,
            "normalize": normalize,
            **kwargs,
        }

        self.features = params["features"]
        self.temperature = params["temperature"]
        self.scale = params["scale"]
        self.normalize = params["normalize"]

        if self.scale is not None and self.normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if self.scale is None:
            self.scale = 2 * math.pi

        if len(boxes.size()) == 2:
            boxes = boxes.unsqueeze(0)

        if self.normalize:
            boxes = boxes / (boxes[:, -1:] + 1e-6) * self.scale

        dim_t = torch.arange(self.features, dtype=torch.float32)

        dim_t = self.temperature ** (2 * self._torch_int_div(dim_t, 2) / self.features)

        # (b, n_t, 4, D//4)
        pos_emb = boxes[:, :, :, None] / dim_t

        pos_emb = torch.stack(
            (pos_emb[:, :, :, 0::2].sin(), pos_emb[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        # (n_t, D)
        pos_emb = pos_emb.squeeze(0).flatten(1)

        return pos_emb

    def _learned_pos_embedding(
        self,
        boxes: torch.Tensor,
        features: int = 1024,
        learn_pos_emb_num: int = 16,
        over_boxes: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Computes learned positional embeddings for boxes using given parameters.

        Args:
            boxes: the input boxes.
            features: Number of features in attention head.
            learn_pos_emb_num: Size of the dictionary of embeddings.
            over_boxes: If True, use box dimensions, rather than box offset and shape.

        Returns:
            torch.Tensor, the learned positional embeddings.
        """
        params = {
            "features": features,
            "learn_pos_emb_num": learn_pos_emb_num,
            "over_boxes": over_boxes,
            **kwargs,
        }

        self.features = params["features"]
        self.learn_pos_emb_num = params["learn_pos_emb_num"]
        self.over_boxes = params["over_boxes"]

        pos_lookup = torch.nn.Embedding(
            self.learn_pos_emb_num * 4, self.features // 4, device=boxes.device
        )

        N = boxes.shape[0]
        boxes = boxes.view(N, 4)

        if self.over_boxes:
            xywh = boxes
        else:
            xywh = torch.cat(
                [(boxes[:, 2:] + boxes[:, :2]) / 2, (boxes[:, 2:] - boxes[:, :2])],
                dim=1,
            )

        l, r, lw, rw = self._compute_weights(xywh, self.learn_pos_emb_num)

        f = pos_lookup.weight.shape[1]

        pos_emb_table = pos_lookup.weight.view(
            self.learn_pos_emb_num, 4, f
        )  # T x 4 x (D * 4)

        pos_le = pos_emb_table.gather(0, l[:, :, None].expand(N, 4, f))  # N x 4 x d
        pos_re = pos_emb_table.gather(0, r[:, :, None].expand(N, 4, f))  # N x 4 x d
        pos_emb = lw[:, :, None] * pos_re + rw[:, :, None] * pos_le

        pos_emb = pos_emb.view(N, 4 * f)

        return pos_emb

    def _learned_temp_embedding(
        self,
        times: torch.Tensor,
        features: int = 1024,
        learn_temp_emb_num: int = 16,
        **kwargs,
    ) -> torch.Tensor:
        """Computes learned temporal embeddings for times using given parameters.

        Args:
            times: the input times.
            features: Number of features in attention head.
            learn_temp_emb_num: Size of the dictionary of embeddings.

        Returns:
            torch.Tensor, the learned temporal embeddings.
        """
        params = {
            "features": features,
            "learn_temp_emb_num": learn_temp_emb_num,
            **kwargs,
        }

        self.features = params["features"]
        self.learn_temp_emb_num = params["learn_temp_emb_num"]

        temp_lookup = torch.nn.Embedding(
            self.learn_temp_emb_num, self.features, device=times.device
        )

        N = times.shape[0]

        l, r, lw, rw = self._compute_weights(times, self.learn_temp_emb_num)

        le = temp_lookup.weight[l]  # T x D --> N x D
        re = temp_lookup.weight[r]

        temp_emb = lw[:, None] * re + rw[:, None] * le

        return temp_emb.view(N, self.features)

    def _compute_weights(
        self, data: torch.Tensor, learn_emb_num: int = 16
    ) -> Tuple[torch.Tensor, ...]:
        """Computes left and right learned embedding weights.

        Args:
            data: the input data (e.g boxes or times).
            learn_temp_emb_num: Size of the dictionary of embeddings.

        Returns:
            A torch.Tensor for each of the left/right indices and weights, respectively
        """
        data = data * learn_emb_num

        left_index = data.clamp(min=0, max=learn_emb_num - 1).long()  # N x 4
        right_index = (
            (left_index + 1).clamp(min=0, max=learn_emb_num - 1).long()
        )  # N x 4

        left_weight = data - left_index.float()  # N x 4

        right_weight = 1.0 - left_weight

        return left_index, right_index, left_weight, right_weight
