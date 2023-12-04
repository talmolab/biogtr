"""Helper functions for visualizing tracking."""
from scipy.interpolate import interp1d
from copy import deepcopy
from tqdm import tqdm
from omegaconf import DictConfig

import seaborn as sns
import imageio
import hydra
import pandas as pd
import numpy as np
import cv2


palette = sns.color_palette("tab10")


def fill_missing(data: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Fill missing values independently along each dimension after the first.

    Args:
        data: the array for which to fill missing value
        kind: How to interpolate missing values using `scipy.interpoloate.interp1d`

    Returns:
        The array with missing values filled in
    """
    # Store initial shape.
    initial_shape = data.shape

    # Flatten after first dim.
    data = data.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(data.shape[-1]):
        y = data[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        data[:, i] = y

    # Restore to initial shape.
    data = data.reshape(initial_shape)

    return data


def annotate_video(
    video,
    labels: pd.DataFrame,
    key: str,
    color_palette=palette,
    trails: bool = True,
    boxes: int = 64,
    names: bool = True,
    centroids: bool = True,
    poses=False,
    save_path: str = "debug_animal",
    fps: int = 30,
) -> list:
    """Annotate video frames with labels.

    Labels video with bboxes, centroids, trajectory trails, and/or poses.

    Args:
        video: The video to be annotated in an ndarray
        labels: The pandas dataframe containing the centroid and/or pose locations of the instances
        key: The key where labels are stored in the dataframe - mostly used for choosing whether to annotate based on pred or gt labels
        color_palette: The matplotlib colorpalette to use for annotating the video. Defaults to `tab10`
        trails: Whether or not to include trajectory history
        boxes: The size of the bbox. If bbox size <= 0 or None then it is not added
        names: Whether or not to annotate with name
        centroids: Whether or not to annotate with
        poses: Whether or not to annotate with poses

    Returns:
        A list of annotated video frames
    """
    writer = imageio.get_writer(save_path, fps=fps)
    color_palette = deepcopy(color_palette)

    if trails:
        track_trails = {}
    try:
        for i in tqdm(sorted(labels["Frame"].unique()), desc="Frame", unit="Frame"):
            frame = video.get_data(i)
            if frame.shape[0] == 1 or frame.shape[-1] == 1:
                frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                frame = (frame * 255).astype(np.uint8).copy()
            lf = labels[labels["Frame"] == i]
            for idx, instance in lf.iterrows():
                if not trails:
                    track_trails = {}

                if poses:
                    # TODO figure out best way to store poses (maybe pass a slp labels file too?)
                    trails = False
                    centroids = False
                    for idx, (pose, edge) in enumerate(
                        zip(instance["poses"], instance["edges"])
                    ):
                        pose = fill_missing(pose.numpy())

                        pred_track_id = instance[key][idx].numpy().tolist()

                        # Add midpt to track trail.
                        if pred_track_id not in list(track_trails.keys()):
                            track_trails[pred_track_id] = []

                        # Select a color based on track_id.
                        track_color_idx = pred_track_id % len(color_palette)
                        track_color = (
                            (np.array(color_palette[track_color_idx]) * 255)
                            .astype(np.uint8)
                            .tolist()[::-1]
                        )

                        for p in pose:
                            # try:
                            #    p = tuple([int(i) for i in p.numpy()][::-1])
                            # except:
                            #    continue

                            p = tuple(int(i) for i in p)[::-1]

                            track_trails[pred_track_id].append(p)

                            frame = cv2.circle(
                                frame, p, radius=2, color=track_color, thickness=-1
                            )

                        for e in edge:
                            source = tuple(int(i) for i in pose[int(e[0])])[::-1]
                            target = tuple(int(i) for i in pose[int(e[1])])[::-1]

                            frame = cv2.line(frame, source, target, track_color, 1)

                if (boxes is not None and boxes > 0) or centroids:
                    # Get coordinates for detected objects in the current frame.
                    x = instance["X"]
                    y = instance["Y"]
                    min_x, min_y, max_x, max_y = (
                        int(x - boxes / 2),
                        int(y - boxes / 2),
                        int(x + boxes / 2),
                        int(y + boxes / 2),
                    )
                    midpt = (int(x), int(y))

                    # print(midpt, type(midpt))

                    # assert idx < len(instance[key])
                    pred_track_id = instance[key]

                    # Add midpt to track trail.
                    if pred_track_id not in list(track_trails.keys()):
                        track_trails[pred_track_id] = []
                    track_trails[pred_track_id].append(midpt)

                    # Select a color based on track_id.
                    track_color_idx = int(pred_track_id) % len(color_palette)
                    track_color = (
                        (np.array(color_palette[track_color_idx]) * 255)
                        .astype(np.uint8)
                        .tolist()[::-1]
                    )

                    # print(instance[key])

                    # Bbox.
                    if boxes is not None and boxes > 0:
                        frame = cv2.rectangle(
                            frame,
                            (min_x, min_y),
                            (max_x, max_y),
                            color=track_color,
                            thickness=2,
                        )

                    # Track trail.
                    if centroids:
                        frame = cv2.circle(
                            frame, midpt, radius=4, color=track_color, thickness=-1
                        )
                        for i in range(0, len(track_trails[pred_track_id]) - 1):
                            frame = cv2.circle(
                                frame,
                                track_trails[pred_track_id][i],
                                radius=4,
                                color=track_color,
                                thickness=-1,
                            )
                            frame = cv2.line(
                                frame,
                                track_trails[pred_track_id][i],
                                track_trails[pred_track_id][i + 1],
                                color=track_color,
                                thickness=2,
                            )

                # Track name.
                if names:
                    frame = cv2.putText(
                        frame,
                        # f"idx:{idx} | track_{pred_track_id}",
                        f"track_{pred_track_id}",
                        org=(int(min_x), max(0, int(min_y) - 10)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.9,
                        color=track_color,
                        thickness=2,
                    )
            writer.append_data(frame)

    except Exception as e:
        writer.close()
        print(e)
        return False

    writer.close()
    return True


def save_vid(
    annotated_frames: list,
    save_path: str = "debug_animal",
    fps: int = 30,
):
    """Save video to file.

    Args:
        annotated_frames: a list of frames annotated by `annotate_frames`
        save_path: The path of the annotated file.
        fps: The frame rate in frames per second of the annotated video
    """
    for idx, (ds_name, data) in enumerate([(save_path, annotated_frames)]):
        imageio.mimwrite(f"{ds_name}.mp4", data, fps=fps, macro_block_size=1)


def color(val: float, thresh: float = 0.01) -> str:
    """Highlight value in dataframe if it is over a threshold.

    Args:
        val: The value to color
        thresh: The threshold for which to color

    Returns:
        A string containing how to highlight the value
    """
    color = "lightblue" if float(val) > thresh else ""
    return f"background-color: {color}"


def bold(val: float, thresh: float = 0.01) -> str:
    """Bold value if it is over a threshold.

    Args:
        val: The value to bold or not
        thresh: The threshold the value has to exceed to be bolded

    Returns:
        A string indicating how to bold the item.
    """
    bold = "bold" if float(val) > thresh else ""
    return f"font-weight: {bold}"


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg: DictConfig):
    """Take in a path to a video + labels file, annotates a video and saves it to the specified path."""
    labels = pd.read_csv(cfg.labels_path)
    video = imageio.get_reader(cfg.vid_path, "ffmpeg")
    annotated_frames = annotate_video(video, labels, **cfg.annotate)
    save_vid(annotated_frames, **cfg.save)


if __name__ == "__main__":
    main()
