"""Module containing helper functions for datasets."""

from PIL import Image
from numpy.typing import ArrayLike
from torchvision.transforms import functional as tvf
from xml.etree import cElementTree as et
import albumentations as A
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sleap_io as sio
import torch


def pad_bbox(bbox: ArrayLike, padding: int = 16) -> torch.Tensor:
    """Pad bounding box coordinates.

    Args:
        bbox: Bounding box in [y1, x1, y2, x2] format.
        padding: Padding to add to each side in pixels.

    Returns:
        Padded bounding box in [y1, x1, y2, x2] format.
    """
    y1, x1, y2, x2 = bbox
    y1, x1 = y1 - padding, x1 - padding
    y2, x2 = y2 + padding, x2 + padding
    return torch.Tensor([y1, x1, y2, x2])


def crop_bbox(img: torch.Tensor, bbox: ArrayLike) -> torch.Tensor:
    """Crop an image to a bounding box.

    Args:
        img: Image as a tensor of shape (channels, height, width).
        bbox: Bounding box in [y1, x1, y2, x2] format.

    Returns:
        Cropped pixels as tensor of shape (channels, height, width).
    """
    # Crop to the bounding box.
    y1, x1, y2, x2 = bbox
    crop = tvf.crop(
        img,
        top=int(y1.round()),
        left=int(x1.round()),
        height=int((y2 - y1).round()),
        width=int((x2 - x1).round()),
    )

    return crop


def get_bbox(center: ArrayLike, size: int | tuple[int]) -> torch.Tensor:
    """Get a square bbox around a centroid coordinates.

    Args:
        center: centroid coordinates in (x,y)
        size: size of the bounding box

    Returns:
        A torch tensor in form y1, x1, y2, x2
    """
    if isinstance(size, int):
        size = (size, size)
    cx, cy = center[0], center[1]

    y1 = max(0, -size[-1] // 2 + cy)
    x1 = max(0, -size[0] // 2 + cx)
    y2 = size[-1] // 2 + cy if y1 != 0 else size[1]
    x2 = size[0] // 2 + cx if x1 != 0 else size[0]
    bbox = torch.Tensor([y1, x1, y2, x2])

    return bbox


def centroid_bbox(points: ArrayLike, anchors: list, crop_size: int) -> torch.Tensor:
    """Calculate bbox around instance centroid.

    This is useful for ensuring that crops are centered around each instance
    in the case of incorrect pose estimates.

    Args:
        points: 2d array of centroid coordinates where each row corresponds to a
            different anchor point.
        anchors: indices of a given anchor point to use as the centroid
        crop_size: Integer specifying the crop height and width

    Returns:
        Bounding box in [y1, x1, y2, x2] format.
    """
    print(anchors)
    for anchor in anchors:
        cx, cy = points[anchor][0], points[anchor][1]
        if not np.isnan(cx):
            break

    bbox = torch.Tensor(
        [
            -crop_size / 2 + cy,
            -crop_size / 2 + cx,
            crop_size / 2 + cy,
            crop_size / 2 + cx,
        ]
    )

    return bbox


def pose_bbox(points: np.ndarray, bbox_size: tuple[int] | int) -> torch.Tensor:
    """Calculate bbox around instance pose.

    Args:
        points: an np array of shape nodes x 2,
        bbox_size: size of bbox either an int indicating square bbox or in (x,y)

    Returns:
        Bounding box in [y1, x1, y2, x2] format.
    """
    if isinstance(bbox_size, int):
        bbox_size = (bbox_size, bbox_size)
    # print(points)

    c = np.nanmean(points, axis=0)
    bbox = torch.Tensor(
        [
            c[-1] - bbox_size[-1] / 2,
            c[0] - bbox_size[0] / 2,
            c[-1] + bbox_size[-1] / 2,
            c[0] + bbox_size[0] / 2,
        ]
    )
    return bbox


def resize_and_pad(img: torch.Tensor, output_size: int) -> torch.Tensor:
    """Resize and pad an image to fit a square output size.

    Args:
        img: Image as a tensor of shape (channels, height, width).
        output_size: Integer size of height and width of output.

    Returns:
        The image zero padded to be of shape (channels, output_size, output_size).
    """
    # Figure out how to scale without breaking aspect ratio.
    img_height, img_width = img.shape[-2:]
    if img_width < img_height:  # taller
        crop_height = output_size
        scale = crop_height / img_height
        crop_width = int(img_width * scale)
    else:  # wider
        crop_width = output_size
        scale = crop_width / img_width
        crop_height = int(img_height * scale)

    # Scale without breaking aspect ratio.
    img = tvf.resize(img, size=[crop_height, crop_width])

    # Pad to square.
    img_height, img_width = img.shape[-2:]
    hp1 = int((output_size - img_width) / 2)
    vp1 = int((output_size - img_height) / 2)
    hp2 = output_size - (img_width + hp1)
    vp2 = output_size - (img_height + vp1)
    padding = (hp1, vp1, hp2, vp2)
    return tvf.pad(img, padding, 0, "constant")


class NodeDropout:
    """Node dropout augmentation.

    Drop up to `n` nodes with probability `p`.
    """

    def __init__(self, p: float, n: int) -> None:
        """Initialize Node Dropout Augmentation.

        Args:
            p: the probability with which to drop the nodes
            n: the maximum number of nodes to drop
        """
        self.n = n
        self.p = p

    def __call__(self, nodes: list[str]) -> list[str]:
        """Wrap `drop_nodes` to enable class call.

        Args:
            nodes: A list of available node names to drop.

        Returns:
            dropped_nodes: A list of up to `self.n` nodes to drop.
        """
        return self.forward(nodes)

    def forward(self, nodes: list[str]) -> list[str]:
        """Drop up to `n` random nodes with probability p.

        Args:
            nodes: A list of available node names to drop.

        Returns:
            dropped_nodes: A list of up to `self.n` nodes to drop.
        """
        if self.n == 0 or self.p == 0:
            return []

        nodes_to_drop = np.random.permutation(nodes)
        node_dropout_p = np.random.uniform(size=len(nodes_to_drop))

        dropped_node_inds = np.where(node_dropout_p < self.p)
        node_dropout_p = node_dropout_p[dropped_node_inds]

        n_nodes_to_drop = min(self.n, len(node_dropout_p))

        dropped_node_inds = np.argpartition(node_dropout_p, -n_nodes_to_drop)[
            -n_nodes_to_drop:
        ]

        dropped_nodes = nodes_to_drop[dropped_node_inds]

        return dropped_nodes


def sorted_anchors(labels: sio.Labels) -> list[str]:
    """Sort anchor names from most instances with that node to least.

    Args:
        labels: a sleap_io.labels object containing all the labels for that video

    Returns:
        A list of anchor names sorted by most nodes to least nodes
    """
    all_anchors = labels.skeletons[0].node_names

    anchor_counts = {anchor: 0 for anchor in all_anchors}

    for i in range(len(labels)):
        lf = labels[i]
        for instance in lf:
            for anchor in all_anchors:
                x, y = instance[anchor].x, instance[anchor].y
                if np.isnan(x) or np.isnan(y):
                    anchor_counts[anchor] += 1

    sorted_anchors = sorted(anchor_counts.keys(), key=lambda k: anchor_counts[k])

    return sorted_anchors


def parse_trackmate(data_path: str) -> pd.DataFrame:
    """Parse trackmate xml or csv labels file.

    Logic adapted from https://github.com/hadim/pytrackmate.

    Args:
        data_path: string path to xml or csv file storing trackmate trajectory labels

    Returns:
        `pandas DataFrame` containing frame number, track_ids,
        and centroid x,y coordinates in pixels
    """
    if data_path.endswith(".xml"):
        root = et.fromstring(open(data_path).read())

        objects = []
        features = root.find("Model").find("FeatureDeclarations").find("SpotFeatures")
        features = [c.get("feature") for c in list(features)] + ["ID"]

        spots = root.find("Model").find("AllSpots")

        objects = []

        for frame in spots.findall("SpotsInFrame"):
            for spot in frame.findall("Spot"):
                single_object = []
                for label in features:
                    single_object.append(spot.get(label))
                objects.append(single_object)

        tracks_df = pd.DataFrame(objects, columns=features)
        tracks_df = tracks_df.astype(np.float)

        filtered_track_ids = [
            int(track.get("TRACK_ID"))
            for track in root.find("Model").find("FilteredTracks").findall("TrackID")
        ]

        label_id = 0
        tracks_df["label"] = np.nan

        tracks = root.find("Model").find("AllTracks")
        for track in tracks.findall("Track"):
            track_id = int(track.get("TRACK_ID"))
            if track_id in filtered_track_ids:
                spot_ids = [
                    (
                        edge.get("SPOT_SOURCE_ID"),
                        edge.get("SPOT_TARGET_ID"),
                        edge.get("EDGE_TIME"),
                    )
                    for edge in track.findall("Edge")
                ]
                spot_ids = np.array(spot_ids).astype("float")[:, :2]
                spot_ids = set(spot_ids.flatten())

                tracks_df.loc[tracks_df["ID"].isin(spot_ids), "TRACK_ID"] = label_id
                label_id += 1

    elif data_path.endswith(".csv"):
        tracks_df = pd.read_csv(data_path, encoding="ISO-8859-1")

    else:
        raise ValueError(f"Unsupported trackmate file extension: {data_path}")

    tracks_df = tracks_df.apply(pd.to_numeric, errors="coerce", downcast="integer")

    posx_key = "POSITION_X"
    posy_key = "POSITION_Y"
    frame_key = "FRAME"
    track_key = "TRACK_ID"

    mapper = {
        "X": posx_key,
        "Y": posy_key,
        "x": posx_key,
        "y": posy_key,
        "Slice n°": frame_key,
        "Track n°": track_key,
    }

    if "t" in tracks_df:
        mapper.update({"t": frame_key})

    tracks_df = tracks_df.rename(mapper=mapper, axis=1)

    if data_path.endswith(".csv"):
        # 0 index track and frame ids
        if min(tracks_df[frame_key]) == 1:
            tracks_df[frame_key] = tracks_df[frame_key] - 1

        if min(tracks_df[track_key] == 1):
            tracks_df[track_key] = tracks_df[track_key] - 1

    return tracks_df


def parse_synthetic(xml_path: str, source: str = "icy") -> pd.DataFrame:
    """Parse .xml labels from synthetic data generated by ICY or ISBI tracking challenge.

    Logic adapted from https://github.com/sylvainprigent/napari-tracks-reader/blob/main/napari_tracks_reader

    Args:
        xml_path: path to .xml file containing ICY or ISBI gt trajectory labels
        source: synthetic dataset type. Should be either icy or isbi

    Returns:
        pandas DataFrame containing frame idx, gt track id
        and centroid x,y coordinates in pixels
    """
    if source.lower() == "icy":
        root_tag = "trackgroup"
    elif source.lower() == "isbi":
        root_tag = "TrackContestISBI2012"
    else:
        raise ValueError(f"{source} source mode not supported")

    tree = et.parse(xml_path)

    root = tree.getroot()
    tracks = np.empty((0, 4))

    # get the trackgroup element
    idx_trackgroup = 0
    for i in range(len(root)):
        if root[i].tag == root_tag:
            idx_trackgroup = i
            break

    ids_map = {}
    track_id = -1
    for track_element in root[idx_trackgroup]:
        track_id += 1

        try:
            ids_map[track_element.attrib["id"]] = track_id
        except:
            pass

        for detection_element in track_element:
            row = [
                float(track_id),
                float(detection_element.attrib["t"]),
                float(detection_element.attrib["y"]),
                float(detection_element.attrib["x"]),
            ]
            tracks = np.concatenate((tracks, [row]), axis=0)

    tracks_df = pd.DataFrame(
        tracks, columns=["TRACK_ID", "FRAME", "POSITION_Y", "POSITION_X"]
    )

    tracks_df = tracks_df.apply(pd.to_numeric, errors="coerce", downcast="integer")

    return tracks_df


class LazyTiffStack:
    """Class used for loading tiffs without loading into memory."""

    def __init__(self, filename: str):
        """Initialize class.

        Args:
            filename: name of tif file to be opened
        """
        # expects spatial, channels
        self.image = Image.open(filename)

    def __getitem__(self, section_idx: int) -> Image:
        """Get frame.

        Args:
            section_idx: index of frame or z-slice to get.

        Returns:
            a PIL image of that frame/z-slice.
        """
        self.image.seek(section_idx)
        return self.image

    def get_section(self, section_idx: int) -> np.array:
        """Get frame as ndarray.

        Args:
            section_idx: index of frame or z-slice to get.

        Returns:
            an np.array of that frame/z-slice.
        """
        section = self.__getitem__(section_idx)
        return np.array(section)

    def close(self):
        """Close tiff stack."""
        self.file.close()


def build_augmentations(augmentations: dict) -> A.Compose:
    """Get augmentations for dataset.

    Args:
        augmentations: a dict containing the name of the augmentations
                       and their parameters

    Returns:
        An Albumentations composition of different augmentations.
    """
    aug_list = []
    for aug_name, aug_args in augmentations.items():
        aug_class = getattr(A, aug_name)
        aug = aug_class(**aug_args)
        aug_list.append(aug)

    augs = A.Compose(
        aug_list,
        p=1.0,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

    return augs


def get_max_padding(height: int, width: int) -> tuple:
    """Calculate maximum padding dimensions for a given height and width.

    Useful if padding is required for rotational augmentations, e.g when
    centroids lie on the borders of an image.

    Args:
        height: The original height.
        width: The original width.

    Returns:
        A tuple containing the padded height and padded width.
    """
    diagonal = math.ceil(math.sqrt(height**2 + width**2))

    padded_height = height + (diagonal - height)
    padded_width = width + (diagonal - width)

    return padded_height, padded_width


def view_training_batch(
    instances: list[dict[str, list[np.ndarray]]], num_frames: int = 1, cmap=None
) -> None:
    """Display a grid of images from a batch of training instances.

    Args:
        instances: A list of training instances, where each instance is a
            dictionary containing the object crops.
        num_frames: The number of frames to display per instance.

    Returns:
        None
    """
    num_crops = len(instances[0]["crops"])
    num_columns = num_crops
    num_rows = num_frames

    base_size = 2
    fig_size = (base_size * num_columns, base_size * num_rows)

    fig, axes = plt.subplots(num_rows, num_columns, figsize=fig_size)

    for i in range(num_frames):
        for j, data in enumerate(instances[i]["crops"]):
            try:
                ax = (
                    axes[j]
                    if num_frames == 1
                    else (axes[i] if num_crops == 1 else axes[i, j])
                )

                (ax.imshow(data.T) if cmap is None else ax.imshow(data.T, cmap=cmap))
                ax.axis("off")

            except Exception as e:
                print(e)
                pass

    plt.tight_layout()
    plt.show()
