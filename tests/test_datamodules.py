from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
# from src.data.scared_datamodule
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
print(root)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def test_scared_datamodule(cfg: DictConfig) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    dm: LightningDataModule = hydra.utils.instantiate(cfg.data)
    print(cfg.data)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    # assert Path(data_dir, "MNIST").exists()
    # assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    # assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    # x, y = batch
    # assert len(x) == batch_size
    # assert len(y) == batch_size
    # assert x.dtype == torch.float32
    # assert y.dtype == torch.int64
    return dm

if __name__ == "__main__":
    test_scared_datamodule()