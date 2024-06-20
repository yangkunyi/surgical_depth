from typing import Any, Dict, Optional, Tuple, List

import torch
import os
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split, Sampler,  SubsetRandomSampler
# from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from src.data.components import MonoDataset
from src.utils import readlines

class SCAREDDataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset.

    The MNIST database of handwritten digits has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a
    fixed-size image. The original black and white images from NIST were size normalized to fit in a 20x20 pixel box
    while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing
    technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of
    mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        split_dir: str = "data/",
        batch_size: int = 12,
        video_length: int = 5,
        height: int = 224,
        width: int = 280,
        num_scales: int = 1,
        num_workers: int = 10,
        pin_memory: bool = True,
    ) -> None:
        """Initialize a `MonoDataset`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        # self.transforms = transforms.Compose(
        #     [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        # )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.frame_idxs = [i - (video_length) + 1 for i in range(video_length)]
        self.batch_size_per_device = batch_size

    # @property
    # def num_classes(self) -> int:
    #     """Get the number of classes.

    #     :return: The number of MNIST classes (10).
    #     """
    #     return 10

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        fpath = os.path.join(self.hparams.split_dir, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        test_filenames = readlines(fpath.format("test"))
        # load and split datasets only if not loaded already
        self.data_train = MonoDataset(
            self.hparams.data_dir, 
            train_filenames, 
            self.hparams.height,
            self.hparams.width,
            self.frame_idxs,
            self.hparams.num_scales,
            True,
        )
        self.data_val = MonoDataset(
            self.hparams.data_dir, 
            val_filenames, 
            self.hparams.height,
            self.hparams.width,
            self.frame_idxs,
            self.hparams.num_scales,
            False,
        )
        self.data_test = MonoDataset(
            self.hparams.data_dir, 
            test_filenames, 
            self.hparams.height,
            self.hparams.width,
            self.frame_idxs,
            self.hparams.num_scales,
            False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        # sampler = FixedBatchesSampler(self.data_train, self.batch_size_per_device, 5)
        # return DataLoader(self.data_train, batch_size=None, sampler=sampler)

        num_batches_per_epoch = 5

        # 3. 计算每个epoch需要的样本数量
        subset_size = num_batches_per_epoch * self.batch_size_per_device

        # 4. 创建随机索引，用于选择子集
        indices = torch.randperm(len(self.data_train))[:subset_size]

        # 5. 创建 SubsetRandomSampler，使用相同的索引但不同的顺序
        sampler = SubsetRandomSampler(indices)
    
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def predict_dataloader(self) -> Any:
        num_batches_per_epoch = 5

        # 3. 计算每个epoch需要的样本数量
        subset_size = num_batches_per_epoch * self.batch_size_per_device

        # 4. 创建随机索引，用于选择子集
        indices = torch.randperm(len(self.data_train))[:subset_size]

        # 5. 创建 SubsetRandomSampler，使用相同的索引但不同的顺序
        sampler = SubsetRandomSampler(indices)
    
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            sampler=sampler,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = SCAREDDataModule()
