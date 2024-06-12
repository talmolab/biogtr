"""Top-level package for dreem."""

import logging.config
from dreem.version import __version__

from dreem.models.global_tracking_transformer import GlobalTrackingTransformer
from dreem.models.gtr_runner import GTRRunner
from dreem.models.transformer import Transformer
from dreem.models.visual_encoder import VisualEncoder

from dreem.io.frame import Frame
from dreem.io.instance import Instance
from dreem.io.association_matrix import AssociationMatrix
from dreem.io.config import Config
from dreem.io.visualize import annotate_video

# from .training import run

from dreem.inference.tracker import Tracker


def setup_logging():
    """Setup logging based on `logging.yaml`."""
    import logging
    import yaml
    import os

    package_directory = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(package_directory, "..", "logging.yaml"), "r") as stream:
        logging_cfg = yaml.load(stream, Loader=yaml.FullLoader)

    logging.config.dictConfig(logging_cfg)
    logger = logging.getLogger("dreem")
