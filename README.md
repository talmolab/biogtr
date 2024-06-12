# DREEM Relates Every Entities' Motion

[![CI](https://github.com/talmolab/dreem/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/dreem/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/dreem/branch/main/graph/badge.svg?token=Sj8kIFl3pi)](https://codecov.io/gh/talmolab/dreem)
[![Documentation](https://img.shields.io/badge/Documentation-dreem.sleap.ai-lightgrey)](https://dreem.sleap.ai)
<!-- [![Release](https://img.shields.io/github/v/release/talmolab/dreem?label=Latest)](https://github.com/talmolab/dreem/releases/)
[![PyPI](https://img.shields.io/pypi/v/dreem?label=PyPI)](https://pypi.org/project/dreem)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dreem) -->

Global Tracking Transformers for biological multi-object tracking.

## Installation
<!-- ### Basic
```
pip install dreem
``` -->
### Development
#### Clone the repository:
```bash
git clone https://github.com/talmolab/dreem && cd dreem
```
#### Set up in a new conda environment:
##### Linux/Windows:
###### GPU-accelerated (requires CUDA/nvidia gpu)
```bash
conda env create -y -f environment.yml && conda activate dreem
```
###### CPU:
```bash
conda env create -y -f environment_cpu.yml && conda activate dreem
```
#### OSX (M chip)
```bash
conda env create -y -f environment_osx-arm.yml && conda activate dreem
```
### Uninstalling
```
conda env remove -n dreem
```

## Usage

Here we describe a basic workflow from setting up data thru training and running inference. Regardless if you're only interested in running inference, we recommend at least skimming thru the entire tutorial as there may be useful information that applies to both training and inference!
### Background

This repo is built on 3 main packages
1. [`hydra`](https://hydra.cc) for config handling
2. [`pytorch`](https://pytorch.org) for model construction
3. [`pytorch-lightning`](https://lightning.ai) for high-level training/eval/inference handling. 

Thus, we recommend skimming through their respective docs to get some familiarity but is not necessary.

### Setup
#### Step 1. Clone the repository:
In your terminal run:
```bash
git clone https://github.com/talmolab/dreem && cd dreem
```
This will clone the dreem repo into your current working directory and then move you inside the `dreem` directory.
#### Step 2. Set up in a new conda environment:
Next run:
```bash
conda env create -y -f environment.yml && conda activate dreem
```
This will create a conda environment called `dreem` which will have `dreem` installed as well as any other dependcies needed to run.
#### Step 3. Activate the `dreem` environment
Finally run:
```bash
conda activate dreem && cd ..
```
This will activate the `dreem` repo and move you back to your original working directory.

### Training

#### Step 1: Generate Ground Truth Data
In order to train a model you need 2 things.
1. A video.
    - For animal data see the [`imageio`](https://imageio.readthedocs.io/en/v2.4.1/formats.html) docs for supported file types.
    - For microscopy data we currently support `.tif` files. Video formats supported by `imageio` coming soon.
2. A ground truth labels file. This labels file contains two main pieces of data:
    1. Detections/instances (i.e. locations of the instances in each frame). This can come in the form of centroids, poses or segmentations
    2. Ground truth identities (also called tracks, or trajectories). These are temporally consistent labels that group detections through time.

##### Step 1.1: Get Initial labels
To generate your initial labels we recommend a couple methods.
 - For animal pose-estimation, we highly recommend heading over to [SLEAP](https://sleap.ai) and running through their work flow. 
 - For microscopy tracking, check out [TrackMate](https://imagej.net/plugins/trackmate/).

 This is because these methods will handle both the detection and tracking steps together. Furthermore, these are currently the two main label formats we support in our data pipelines for arbitrary [animal](dreem/datasets/sleap_dataset.py) and [microscopy](dreem/datasets/microscopy_dataset.py) datasets. If you'd like to use a different method, (e.g DeepLabCut or ilastik etc), the easiest way to make your data compatible with `dreem` is to convert your labels to a `.slp` file and your video to an [`imageio`-supported]((https://imageio.readthedocs.io/en/v2.4.1/formats.html)) video format. See the next section for more information on how to do this. Alternatively, you can write a custom dataloader but that will take significantly more overhead.
##### Step 1.2 Proofreading
Once you have your labels file containing the detections and tracks, you'll want to make sure to proofread your labels. This is because object-tracking is hard and the methods we recommend above may have made mistakes. The most important thing to train a good model is to have high quality data. In our case good quality means two things:

1. No identity switches. This is essential for training our object tracking model. If there are identity switches (say an animal/cell/organelle in frame 1 has track label 0 but in frame 2 it has track label 1), this will basically teach the model to make mistakes and cause irrecoverable failures. See below for an example of an identity switch.
//TODO add example id switch
2. Good detections. Because the input to the model is a crop centered around each detection, we want to make sure the coordinates we crop around are as accurate as possible. This is a bit more arbitrary and up to the user to determine what is a "good" detection. A general rule of thumb is to avoid having detection coordinates be hard to distinguish. For instance, with animal pose estimation, we want to avoid having the key points on two instances. For segmentation, the boundaries should be as tight as possible. This may be unavoidable however, in cases of occlusion and overlap. See below for example cases of good and bad detections.
//TODO add example good vs bad detection.

We recommend using the [`sleap-label` gui](https://sleap.ai/guides/gui.html) for [proofreading](https://sleap.ai/guides/proofreading.html#id1). This is because SLEAP's in-built gui provides useful. functionality for visualizing detections, moving them, and reassigning tracks. It also provides some nice heuristics for flagging where switches may have occured. 

###### Converting data to a SLEAP compatible format.
In order to use the SLEAP gui you'll need to have your labels and videos in a SLEAP comptabile format. Check out [the sleap-io docs](https://io.sleap.ai/latest/formats/) for available formats. The easiest way to ensure your labels are compatible with sleap is to convert them to a `.slp` file. For animal tracking, if you used SLEAP, this is already how SLEAP saves your labesl so you can start proofreading right away. Otherwise if you used a different system (e.g DeepLabCut) check out [`sleap.io.convert`](https://sleap.ai/api/sleap.io.convert.html#module-sleap.io.convert) for available converters. With microscopy, we highly recommend starting out with TrackMate and then proofread in SLEAP's gui. Here is [a converter from trackmate's output to a `.slp`](https://gist.github.com/aaprasad/5243be0785a40e9dafa1697ce2258e3e) file. In general, you can use [`sleap-io`](https://io.sleap.ai/latest/) to write a custom converter to `.slp` if you'd like to use the sleap-gui for proofreading. Once you've ensured that your labels files have no identity switches and your detections are as good as you're willing to make them, they're ready to use for training.

##### Step 1.3 Organize data.

Although, our data loading does take paths to label files and video files directly so it's fairly flexible, we recommend organizing your data with a couple things in mind.

1. Match video and labels file stems. Because our dataloaders just take in a list of video files and a list of corresponding label files, the order in which the corresponding files are passed must match. (e.g `[file1.slp, file2.slp], [vid1.mp4, vid2.mp4]`). In order to make programmatic file searching easy its best to save your labels and vid files with the same stem so that you can ensure the ordering will be consistent. Its also just best practice so you know which video a labels file corresponds to.
2. Store corresponding videos and labels files in the same directory. This will again make searching for the files much easier. 

The best practice is to have your dataset organized as follows:
```
dataset_name/
    train/
        vid_1.{VID_EXTENSION}
        vid_1.{slp, csv, xml} #depends on labels source
            .
            .
            .
        vid_n.{VID_EXTENSION}
        vid_n.{slp, csv, xml}
    val/
        vid_1.{VID_EXTENSION}
        vid_1.{slp, csv, xml}
            .
            .
            .
        vid_n.{VID_EXTENSION}
        vid_n.{slp, csv, xml}
    test/
        vid_1.{VID_EXTENSION}
        vid_1.{slp, csv, xml}
            .
            .
            .
        vid_n.{slp, csv, xml}
        vid_n.{slp, csv, xml}
```

#### Step 2. Training

Now that you have your dataset set up, we can start training! In [`dreem.training.train`](https://github.com/talmolab/dreem/blob/main/dreem/training/train.py), we provide a train script that allows you to train with just a `.yaml` file. It also serves as a good example for how to write your own train script if you're an advanced user and would like to have some additional flexibility!

##### Step 2.1. Setup Config:

The input into our training script is a `.yaml` file that contains all the parameters needed for training. Please checkout the [`README`](dreem/training/configs/README.md) in `dreem/training/configs` for a detailed description of all the parameters and how to set up the config. We also provide an example config in [`dreem/training/configs/base.yaml`](dreem/training/configs/base.yaml) to give you an idea for how the config should look. In general, the best practice is to keep a single `base.yaml` file which has all the default arguments you'd like to use. Then you can have a second `.yaml` file which will override only those specific set of parameters when training (for an example see [`dreem/training/configs/params.yaml`](dreem/training/configs/params.yaml)).

##### Step 2.2 Train Model

Once you have your config file and dataset set up, training is as easy as running

```bash
dreem-train --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]
```
where `CONFIG_DIR` is the directory that `hydra` should search for the `config.yaml` and `CONFIG_STEM` is the name of the config without the `.yaml` extension.

e.g. If I have a config file called `base.yaml` inside my `/home/aaprasad/dreem_configs` directory I can call
```bash
dreem-train --config-base=/home/aaprasad/dreem_configs --config-name=base
```

> Note: you can use relative paths as well but may be a bit riskier so we recommend absolute paths whenever possible.

###### Overriding Arguments
Instead of changing the `base.yaml` file every time you want to run a different config, `hydra` enables us to either 
1. provide another `.yaml` file with a subset of the parameters to overide
2. provide the args to the cli directly

###### File-based override
For overriding specific params with a sub-config, you can specify a `params_config` key and path in your `config.yaml` or run

```bash
dreem-train --config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM] ++params_config="/path/to/params.yaml"
```

e.g. If I have a `params_to_override.yaml` file inside my `/home/aaprasad/dreem_configs` directory that contains a only a small selection of parameters that I'd like to override, I can run:

```bash
dreem-train --config-base=/home/aaprasad/dreem_configs --config-name=base ++params_config=/home/aaprasad/dreem_configs/params_to_override.yaml
```

##### CLI-based override
For directly overriding a specific param via the command line directly you can use the `section.param=key` syntax as so:

```bash
dreem-train --config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM] section.param=value
```

e.g If now I want to override a couple parameters again, say change the number of attention heads and change the name of this run in my logger, I can pass `model.head=3` and `logger.name="test_nheads=3"` into 

```bash
dreem-train --config-base=/home/aaprasad/dreem_configs --config-name=base model.nhead=3 logger.name="test_nheads=3"
```
> Note: using the `section.param` syntax for CLI override will only override if the parameter exists in your config file otherwise an error will be thrown
> if you'd like add a new parameter you can add `++` to the front of `section.param` e.g `++model.nhead=3`, However in this case, if the parameter exists in the config it will throw an error.
> thus if you'd like to overwrite when the parameter exists, otherwise create a new one you can add a single `+` to the front of `section.param` e.g `+model.nhead=3`. 
>When using `+`, you'll want to make sure you've matched the param exactly, otherwise it will simply add a new parameter with your spelling and the original value won't change. 
> e.g doing `model.n_head=3` will cause the original `model.nhead` to stay the same and just add a `model.n_head` field.

See [here](https://hydra.cc/docs/advanced/override_grammar/basic/) for more information on overriding params.

> Note: You will usually only need the direct override or the file-based override however you can technically do both via 
> ```bash
> dreem-train --config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM] ++params_config="/path/to/params.yaml" section.param=value
> ```
> e.g. alongside the `params_to_override.yaml`
> ```bash
> dreem-train --config-base=/home/aaprasad/dreem_configs --config-name=base ++params_config=/home/aaprasad/dreem_configs/params_to_override.yaml model.nhead=3 logger.name="test_nheads=3"
> ```
> but the `--config-base=[CONFIG_DIR] --config-name=[BASE_CONFIG_STEM]` arguments are always required. 
> However, be careful to ensure there are no conflicts when combining CLI and file-based override syntax
> (e.g make sure `section.param` doesn't appear both in your override file and in the CLI). 
> This is because hydra does not make a distinction, it will first overwrite with the CLI value and then overwrite with the params file value.
> this means that it will always default to the params file value when there are conflicts.

#### Output
The output of the train script will be at least 1 `*.ckpt` file, assuming you've configured the `checkpointing` section of the config correctly and depending on the params you've used.

### Inference

#### Step 1. Setup Data
##### Step 1.1 Get detections
One of the main advantages of this system is that we *decouple* detection and tracking so you can use off-the-shelf high performant detectors/pose-estimators/segementors. This means that in order to run inference(tracking) with our model you need 3 things.
1. A pretrained model saved as a `.ckpt` file.
2. A video.
    - For animal data see the [`imageio`](https://imageio.readthedocs.io/en/v2.4.1/formats.html) docs for supported file types.
    - For microscopy data we currently support `.tif` files. Video formats supported by `imageio` coming soon.
3. A labels file. This labels file is slightly different than in training because we only need detections for tracking since the tracks will of course come from our model predictions.

We still recommend using SLEAP and TrackMate to generate these labels however this time you would only need to proofread your detections if you'd like to. For TrackMate we recommend using the "spots table" labels `.csv` instead of the "tracks table" in the case where trackmate misses tracks.
##### Step 1.2 Organize Data

We recommend setting up your inference data with the same practices we used for training.
1. Match video and labels file stems. Because our dataloaders just take in a list of video files and a list of corresponding label files, the order in which the corresponding files are passed must match. (e.g `[file1.slp, file2.slp], [vid1.mp4, vid2.mp4]`). In order to make programmatic file searching easy its best to save your labels and vid files with the same stem so that you can ensure the ordering will be consistent. Its also just best practice so you know which video a labels file corresponds to.
2. Store corresponding videos and labels files in the same directory. This will again make searching for the files much easier. 

The best practice is to have your dataset organized as follows:
```
dataset_name/
    inference/
        vid_1.{VID_EXTENSION}
        vid_1.{slp, csv, xml} #extension depends on labels source, doesn't need tracks
            .
            .
            .
        vid_n.{slp, csv, xml}
        vid_n.{slp, csv, xml}
```

#### Step 2. Set up config

Similar to training, we need to set up a config file that specifies 
1. a `ckpt_path`
2. a `out_dir`
3. a `Tracker` config
4. a `dataset` config with a `test_dataset` subsection containing dataset params.

Please see the [README](dreem/inference/configs/inference.yaml) in `dreem/inference/configs` for a walk through of the inference params as well as how to set up an inference conifg and see [`dreem/inference/configs/inference.yaml`](dreem/inference/configs/inference.yaml) for an example inference config file.

#### Step 3 Run Inference

Just like training we can use the hydra syntax for specifying arguments via the cli. Thus you can run inference via:

```bash
dreem-track --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM]
```

e.g. If I had an inference config called `track.yaml` inside `/home/aaprasad/dreem_configs` 

```bash
dreem-track --config-base=/home/aaprasad/dreem_configs --config-name=track
```

##### Overriding Parameters.
Because there aren't as many parameters during inference as during training we recommend just using the cli-based override rather than the file based but you're more than welcome to do so.

In order to override params via the CLI, we can use the same `hydra` `section.param` syntax:

```bash
dreem-track --config-base=[CONFIG_DIR] --config-name=[CONFIG_STEM] section.param=[VALUE]
```
e.g if I want to set the window size of the tracker to 32 instead of 8 through `tracker.window_size=32` and use a different model saved in `/home/aaprasad/models/new_best.ckpt` I can do:
```bash
dreem-track --config-base=/home/aaprasad/dreem_configs --config-name=track ckpt_path="/home/aaprasad/models/new_best.ckpt" tracker.window_size=32`
```
#### Output
This will run inference on the videos/detections you specified in the `dataset.test_dataset` section of the config and save the tracks to individual `[VID_NAME].dreem_inference.slp` files. If an `outdir` is specified in the config it will save to  `./[OUTDIR]/[VID_NAME].dreem_inference.slp`, otherwise it will just save to `./results/[VID_NAME].dreem_inference.slp`. Now you can load the file with `sleap-io` and do what you please!
