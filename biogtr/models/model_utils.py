"""Module containing model helper functions."""
from copy import deepcopy
from typing import List, Tuple, Iterable
from pytorch_lightning import loggers
from biogtr.data_structures import Frame
import torch


def get_boxes_times(frames: List[Frame]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the bounding boxes and frame indices from the input list of instances.

    Args:
        instances (List[Dict]): List of instance dictionaries

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors containing the
                                            bounding boxes and corresponding frame
                                            indices, respectively.
    """
    boxes, times = [], []
    _, h, w = frames[0].img_shape.flatten()

    for fidx, frame in enumerate(frames):
        bbox = deepcopy(frame.get_bboxes())
        bbox[:, [0, 2]] /= w
        bbox[:, [1, 3]] /= h

        boxes.append(bbox)
        times.append(torch.full((bbox.shape[0],), fidx))

    boxes = torch.cat(boxes, dim=0)  # N x 4
    times = torch.cat(times, dim=0).to(boxes.device)  # N
    return boxes, times


def softmax_asso(asso_output: list[torch.Tensor]) -> list[torch.Tensor]:
    """Apply the softmax activation function on asso_output.

    Args:
        asso_output: Raw logits output of the tracking transformer. A list of
            torch tensors of shape (T, N_t, N_i) where:
                T: the length of the window
                N_t: number of instances in current/query frame (rightmost frame
                    of the window).
                N_i: number of detected instances in i-th frame of window.

    Returns:
        asso_output: Probabilities following softmax function, with same shape
            as input.
    """
    asso_active = []
    for asso in asso_output:
        asso = torch.cat([asso, asso.new_zeros((asso.shape[0], 1))], dim=1).softmax(
            dim=1
        )[:, :-1]
        asso_active.append(asso)

    return asso_active


def init_optimizer(params: Iterable, config: dict):
    """Initialize optimizer based on config parameters.

    Allows more flexibility in which optimizer to use

    Args:
        params: model parameters to be optimized
        config: optimizer hyperparameters including optimizer name

    Returns:
        optimizer: A torch.Optimizer with specified params
    """
    optimizer = config["name"]
    optimizer_params = {
        param: val for param, val in config.items() if param.lower() != "name"
    }

    try:
        optimizer_class = getattr(torch.optim, optimizer)
    except AttributeError:
        if optimizer_class is None:
            print(
                f"Couldn't instantiate {optimizer} as given. Trying with capitalization"
            )
            optimizer_class = getattr(torch.optim, optimizer.lower().capitalize())
        if optimizer_class is None:
            print(
                f"Couldnt instantiate {optimizer} with capitalization, Final attempt with all caps"
            )
            optimizer_class = getattr(torch.optim, optimizer.upper(), None)

    if optimizer_class is None:
        raise ValueError(f"Unsupported optimizer type: {optimizer}")

    return optimizer_class(params, **optimizer_params)


def init_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    """Initialize scheduler based on config parameters.

    Allows more flexibility in choosing which scheduler to use.

    Args:
        optimizer: optimizer for which to adjust lr
        config: lr scheduler hyperparameters including scheduler name

    Returns:
        scheduler: A scheduler with specified params
    """
    scheduler = config["name"]
    scheduler_params = {
        param: val for param, val in config.items() if param.lower() != "name"
    }
    try:
        scheduler_class = getattr(torch.optim.lr_scheduler, scheduler)
    except AttributeError:
        if scheduler_class is None:
            print(
                f"Couldn't instantiate {scheduler} as given. Trying with capitalization"
            )
            scheduler_class = getattr(
                torch.optim.lr_scheduler, scheduler.lower().capitalize()
            )
        if scheduler_class is None:
            print(
                f"Couldnt instantiate {scheduler} with capitalization, Final attempt with all caps"
            )
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler.upper(), None)

    if scheduler_class is None:
        raise ValueError(f"Unsupported optimizer type: {scheduler}")

    return scheduler_class(optimizer, **scheduler_params)


def init_logger(config: dict):
    """Initialize logger based on config parameters.

    Allows more flexibility in choosing which logger to use.

    Args:
        config: logger hyperparameters

    Returns:
        logger: A logger with specified params (or None).
    """
    logger_type = config.pop("logger_type", None)

    valid_loggers = [
        "CSVLogger",
        "TensorBoardLogger",
        "WandbLogger",
    ]

    if logger_type in valid_loggers:
        logger_class = getattr(loggers, logger_type)
        try:
            return logger_class(**config)
        except Exception as e:
            print(e, logger_type)
    else:
        print(
            f"{logger_type} not one of {valid_loggers} or set to None, skipping logging"
        )
        return None
