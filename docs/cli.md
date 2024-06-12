# Command-line Interface

DREEM provides several types of functionality accessible through a command prompt.

## Training

### `dreem-train`

`dreem-train` is the command-line interface for [training](./reference/dreem/training/train.md). Use this for training on a remote machine/cluster/colab notebook instead of using the API directly.

#### Usage

```
usage: dreem-train [-h] [--hydra-help] [--config-dir] [--config-name] [+params_config] [+batch_config]

positional arguments:
    --config-dir    Path to configs dir
    --config-name   Name of the .yaml config file stored in config-dir without the .yaml extension

optional arguments:
    -h, --help      Shows the application's help and exit.
    --hydra-help    Shows Hydra specific flags (recommended over -h) 
    +params_config  Path to .yaml file containing subset of params to override
    +batch_config   Path to .csv file where each row indicates params to override for a single task in a batch job
```

See [the usage guide](./usage.md#step-21-setup-config) for a more in-depth explanation on how to use `dreem-train` and see [the training config walkthrough](./configs/training.md) for all available parameters.

### `dreem-track`

`dreem-track` is the command-line interface for [inference](./reference/dreem/inference/track.md). Use this for tracking using a pretrained model on a remote machine/cluster/notebook instead of using the API directly.

#### Usage

```
usage: dreem-track [-h] [--hydra-help] [--config-dir] [--config-name] [+params_config] [+batch_config]

positional arguments:
    --config-dir    Path to configs dir
    --config-name   Name of the .yaml config file stored in config-dir without the .yaml extension

optional arguments:
    -h, --help      Shows the application's help and exit.
    --hydra-help    Shows Hydra specific flags (recommended over -h) 
    +params_config  Path to .yaml file containing subset of params to override
    +batch_config   Path to .csv file where each row indicates params to override for a single task in a batch job
```
See [the usage guide](./usage.md#step-2-set-up-config) for a more in-depth explanation on how to use `dreem-track` and see [the inference config walkthrough](./configs/inference.md) for all available parameters.

### `dreem-visualize`

`dreem-visualize` is the command-line utility for [generating annotated videos](./reference/dreem/io/visualize.md#dreem.io.visualize.annotate_video) based on DREEM output

#### Usage

```
usage: dreem-visualize [-h] [--hydra-help] [+labels_path] [+vid_path] [+save_path] [+annotate.key] [--config-path] [+annotate.color_palette="tab20"] [+annotate.trails=2] [+annotate.boxes=64] [+annotate.names=True] [+annotate.track_scores=0.5] [+annotate.centroids=4] [+annotate.poses=False] [+annotate.fps=30] [+annotate.alpha=0.2]

positional arguments:
    +labels_path                The path to the dreem tracks as a `.csv` file containing the following keys:
                                    - "X": the X position of the instance in pixels
                                    - "Y": the Y position of the instance in pixels
                                    - "Frame_id": The frame of the video in which the instance occurs
                                    - "Pred_track_id": The track id output from DREEM
                                    - "Track_score": The trajectory score output from DREEM
                                    - (Optional) "Gt_track_id": the gt track id for the instance (mostly used for debugging)
                                where each row is an instance. See `dreem.inference.track.export_trajectories` 
    +vid_path                   The path to the video file in an `imageio`-accepted format (See: https://imageio.readthedocs.io/en/v2.4.1/formats.html)
    +save_path                  The path to where you want to store the annotated video (e.g "./tracked_videos/dreem_output.mp4")
    +annotate.key               The key where labels are stored in the dataframe - mostly used for choosing whether to annotate based on pred or gt labels

optional arguments
    -h, --help                  Shows the application's help and exit.
    --hydra-help                Shows Hydra specific flags (recommended over -h) 
    --config-path               The path to the config .yaml file containing params for annotating. We recommend using this file in lieu of any "annotate.*" arguments.
    +annotate.color_palette     The matplotlib colorpalette to use for annotating the video. Defaults to `tab10`
    +annotate.trails            The size of the track trails. If the trail size <=0 or none then it is not used.
    +annotate.boxes             The size of the bbox. If bbox size <= 0 or None then it is not added
    +annotate.names             Whether or not to annotate with name
    +annotate.centroids         The size of the centroid. If centroid size <=0 or None then it is not added.
    +annotate.poses             Whether or not to annotate with poses
    +annotate.fps               The frames-per-second of the annotated video
    +annotate.alpha             The opacity of the centroid annotations.
```

As noted above, instead of specifying each individual `+annotate.*` param we recommend setting up a visualize config (e.g `./configs/visualize.yaml`) which looks like:

```YAML
annotate:
    color_palette: "tab10"
    trails: 5
    boxes: (64, 64)
    names: true
    centroids: 10
    poses: false
    fps: 30
    alpha: 0.2
```

and then running:

```bash
dreem-visualize +labels_path="/path/to/labels.csv" +vid_path="/path/to/vid.{VID EXTENSION} +annotate.key="Pred_track_id" --config-path="/path/to/configs/visualize.yaml"
```