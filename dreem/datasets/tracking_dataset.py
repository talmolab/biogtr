"""Module containing Lightning module wrapper around all other datasets."""

from dreem.datasets.cell_tracking_dataset import CellTrackingDataset
from dreem.datasets.microscopy_dataset import MicroscopyDataset
from dreem.datasets.sleap_dataset import SleapDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch


"""
Lightning wrapper for tracking datasets
"""


class TrackingDataset(LightningDataModule):
    """Lightning dataset used to load dataloaders for train, test and validation.

    Nice for wrapping around other data formats.
    """

    def __init__(
        self,
        train_ds: SleapDataset | MicroscopyDataset | CellTrackingDataset | None = None,
        train_dl: DataLoader | None = None,
        val_ds: SleapDataset | MicroscopyDataset | CellTrackingDataset | None = None,
        val_dl: DataLoader | None = None,
        test_ds: SleapDataset | MicroscopyDataset | CellTrackingDataset | None = None,
        test_dl: DataLoader | None = None,
    ):
        """Initialize tracking dataset.

        Args:
            train_ds: Sleap or Microscopy training Dataset
            train_dl: Training dataloader. Only used for overriding `train_dataloader`.
            val_ds: Sleap or Microscopy Validation set
            val_dl : Validation dataloader. Only used for overriding `val_dataloader`.
            test_ds: Sleap or Microscopy test set
            test_dl : Test dataloader. Only used for overriding `test_dataloader`.
        """
        super().__init__()
        self.train_ds = train_ds
        self.train_dl = train_dl
        self.val_ds = val_ds
        self.val_dl = val_dl
        self.test_ds = test_ds
        self.test_dl = test_dl

    def setup(self, stage=None):
        """Set up lightning dataset.

        UNUSED.
        """
        pass

    def train_dataloader(self) -> DataLoader:
        """Get train_dataloader.

        Returns: The Training Dataloader.
        """
        if self.train_dl is None and self.train_ds is None:
            return None
        elif self.train_dl is None:
            return DataLoader(
                self.train_ds,
                batch_size=1,
                shuffle=True,
                pin_memory=False,
                collate_fn=self.train_ds.no_batching_fn,
                num_workers=0,
                generator=(
                    torch.Generator(device="cuda")
                    if torch.cuda.is_available()
                    else torch.Generator()
                ),
            )
        else:
            return self.train_dl

    def val_dataloader(self) -> DataLoader:
        """Get val dataloader.

        Returns: The validation dataloader.
        """
        if self.val_dl is None and self.val_ds is None:
            return None
        elif self.val_dl is None:
            return DataLoader(
                self.val_ds,
                batch_size=1,
                shuffle=False,
                pin_memory=0,
                collate_fn=self.train_ds.no_batching_fn,
                num_workers=False,
                generator=None,
            )
        else:
            return self.val_dl

    def test_dataloader(self) -> DataLoader:
        """Get.

        Returns: The test dataloader
        """
        if self.test_dl is None and self.test_ds is None:
            return None
        elif self.test_dl is None:
            return DataLoader(
                self.test_ds,
                batch_size=1,
                shuffle=False,
                pin_memory=0,
                collate_fn=self.train_ds.no_batching_fn,
                num_workers=False,
                generator=None,
            )
        else:
            return self.test_dl
