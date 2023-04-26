import torch
import torch
import pandas as pd
import data_utils
from torchvision.transforms import functional as tvf
from skimage.io import imread
from PIL import Image
from torch.utils import Dataset
from parsers import parse_trackmate, parse_ICY, parse_ISBI


class MicroscopyDataset(Dataset):
    def __init__(
        self,
        videos,
        tracks,
        parser,
        padding=5,
        crop_size=20,
        chunk=False,
        clip_length=10,
        mode="Train",
        tfm=None,
        tfm_cfg=None,
    ):
        """
        Dataset for loading Microscopy Data
        Args:
            videos: paths to raw microscopy videos
            tracks: paths to trackmate gt labels (either .xml or .csv)
            parser: parsing function from biogtr.data.parsers to parse annotation files
            from different file types and get them to a common format
            padding: amount of padding around object crops
            crop_size: the size of the object crops
            chunk: whether or not to chunk the dataset into batches
            clip_length: the number of frames in each chunk
            mode: `train` or `val`. Determines whether this dataset is used for training or validation.
            Currently doesn't affect dataset logic
            tfm: The augmentation function from albumentations
            tfm_cfg: The config for the augmentations
        """
        self.videos = videos
        self.tracks = tracks
        self.parse = parser
        self.chunk = chunk
        self.clip_length = clip_length
        self.crop_size = crop_size
        self.padding = padding
        self.mode = mode
        self.tfm = tfm
        self.tfm_cfg = tfm_cfg

        self.labels = [
            self.parse(self.tracks[video_idx])
            for video_idx in torch.arange(len(self.tracks))
        ]

        self.frame_idx = [
            torch.arange(Image.open(video).n_frames)
            if type(video) == str
            else torch.arange(len(video))
            for video in self.videos
        ]

        if self.chunk:
            self.chunks = [
                [i * self.clip_length for i in range(len(video) // self.clip_length)]
                for video in self.frame_idx
            ]

            self.chunked_frame_idx, self.video_idx = [], []
            for i, (split, frame_idx) in enumerate(zip(self.chunks, self.frame_idx)):
                frame_idx_split = torch.split(frame_idx, split)[1:]
                self.chunked_frame_idx.extend(frame_idx_split)
                self.video_idx.extend(len(frame_idx_split) * [i])
        else:
            self.chunked_frame_idx = self.frame_idx
            self.video_idx = [i for i in range(len(self.videos))]

    def __len__(self):
        """
        Get the size of the dataset
        Returns the size or the number of chunks in the dataset
        """
        return len(self.chunked_frame_idx)

    def no_batching_fn(self, batch):
        return batch

    def __getitem__(self, idx):
        """
        Get an element of the dataset
        Returns a list of dicts where each dict corresponds a frame in the chunk and each value is a `torch.Tensor`
        Dict Elements:
        {
                    "video_id": The video being passed through the transformer,
                    "img_shape": the shape of each frame,
                    "frame_id": the specific frame in the entire video being used,
                    "num_detected": The number of objects in the frame,
                    "gt_track_ids": The ground truth labels,
                    "bboxes": The bounding boxes of each object,
                    "crops": The raw pixel crops,
                    "features": The feature vectors for each crop outputed by the CNN encoder,
                    "pred_track_ids": The predicted trajectory labels from the tracker,
                    "asso_output": the association matrix preprocessing,
                    "matches": the true positives from the model,
                    "traj_score": the association matrix post processing,
            }
            Args:
                idx: the index of the batch. Note this is not the index of the video or the frame.
        """
        video_idx = self.video_idx[idx]
        frame_idx = self.chunked_frame_idx[idx]
        track = self.labels[idx]
        if type(self.videos[video_idx]) == list:
            video = self.videos[video_idx]
            video = torch.stack([torch.Tensor(imread(video[i])) for i in frame_idx])
            print(len(self.videos[video_idx]))
            print(video.shape)
        else:
            video = imread(self.videos[video_idx])
        if self.tfm is not None and self.tfm_cfg is not None:
            aug = self.tfm(**self.frm_cfg)
        elif self.tfm is not None and self.tfm_cfg is None:
            aug = self.tfm
        else:
            aug = None
        instances = []

        for i in frame_idx:
            gt_track_ids, centroids, bboxes, crops = [], [], [], []
            img = video[i]
            # padded = np.pad(sec, pad_width=50, mode="constant", constant_values=0)
            lf = track.loc[track["FRAME"] == i]
            for instance in sorted(lf["TRACK_ID"].unique()):
                # try:
                gt_track_ids.append(int(instance))

                x = lf[lf["TRACK_ID"] == instance]["POSITION_X"].iloc[0]
                y = lf[lf["TRACK_ID"] == instance]["POSITION_Y"].iloc[0]
                centroids.append(torch.Tensor([x, y]).astype("float32"))
            if aug is not None:
                augmented = aug(image=img, keypoints=torch.vstack(centroids))
                img, centroids = augmented["image"], augmented["keypoints"]
            img = tvf.to_tensor(img)

            for c in centroids:
                bbox = data_utils.pad_bbox(
                    data_utils.get_bbox([int(c[0]), int(c[1])], self.crop_size),
                    padding=self.padding,
                )
                bboxes.append(bbox)

            for bbox in bboxes:
                crop = data_utils.crop_bbox(img, bbox)
                crops.append(crop)

            #     bb = self._create_bb([int(x), int(y)], self.crop_size, padding=50)
            #     # print(bb)
            #     # todo: we should definitely fix this, very confusing
            #     min_y, min_x, max_y, max_x = bb
            #     bboxes.append([min_x, min_y, max_x, max_y])

            #     crop = padded[bb[0] : bb[2], bb[1] : bb[3]]

            #     crops.append(tvf.to_tensor(crop))
            #     # except Exception as e:
            #     #     print(e)
            #     #     break
            # # transformer was updated to take _,h,w
            # padded = np.expand_dims(padded, axis=0)
            instances.append(
                {
                    "video_id": torch.Tensor([video_idx]),
                    "img_shape": torch.Tensor([img.shape]),
                    "frame_id": torch.Tensor([i]),
                    "num_detected": torch.Tensor([len(bboxes)]),
                    "gt_track_ids": torch.Tensor(gt_track_ids).type(torch.int64),
                    "bboxes": torch.Tensor(bboxes),
                    "crops": torch.stack(crops),
                    "features": torch.Tensor([]),
                    "pred_track_ids": torch.Tensor([-1 for _ in range(len(bboxes))]),
                    "asso_output": torch.Tensor([]),
                    "matches": torch.Tensor([]),
                    "traj_score": torch.Tensor([]),
                }
            )
        return instances
