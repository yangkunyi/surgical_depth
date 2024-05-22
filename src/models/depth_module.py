from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from .components import R2Plus1D, DinoEncoder, DPTHead
from src.utils import compute_errors
import torch.nn.functional as F
import time

MAX_DEPTH = 150
MIN_DEPTH = 1e-3

class DepthLitModule(LightningModule):
    """ a `LightningModule` for depth estimation.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.
·
    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        alpha: float,
        beta: float,
        compile: bool,
    ) -> None:
        """Initialize a `DepthLitModule`.

        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param alpha: The weight of feature loss
        :param beta: The weight of depth loss
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # self.start_time = time.time()

        self.model = {}
        self.temporal_encoder = R2Plus1D([2,2,2,2])
        self.dino_encoder = DinoEncoder()
        self.depth_decoder = DPTHead(1)
        self.dino_encoder.requires_grad_(False)
        # loss function
        self.criterion = torch.nn.MSELoss()
        self.feature_loss = MeanMetric()
        # # metric objects for calculating and averaging accuracy across batches
        # self.train_acc = Accuracy(task="multiclass", num_classes=10)
        # self.val_acc = Accuracy(task="multiclass", num_classes=10)
        # self.test_acc = Accuracy(task="multiclass", num_classes=10)

        # # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_loss_gt = MeanMetric()
        self.train_loss_feature = MeanMetric()
        self.val_loss = MeanMetric()
        

        self.test_loss = MeanMetric()
        self.errors = [MeanMetric() for _ in range(7)]
        
        # # for tracking best so far validation accuracy
        # self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model.

        :param x: A tensor of images.
        """

        return self.model['depth_decoder'](self.model['temporal_encoder'](x))

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        # self.val_loss.reset()
        # self.val_acc.reset()
        # self.val_acc_best.reset()
        pass

    def model_step(
        self, 
        batch
    ) :
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        """
        
        self.feature_loss.reset()


        patch_h = batch['color_clip'].shape[-2] // 14
        patch_w = batch['color_clip'].shape[-1] // 14

        cnn_feature = self.temporal_encoder(batch['color_clip'])
        dino_feature = self.dino_encoder.temporal_forward(batch['color_clip'])
        pred_depth = self.depth_decoder.temporal_forward(cnn_feature, patch_h, patch_w)
        gt_depth = batch['depth_gt_clip']

        for i in range(len(cnn_feature)):
            self.feature_loss.update(self.hparams.alpha * self.criterion(cnn_feature[i], dino_feature[i]))
        
        pred_depth = F.interpolate(pred_depth, size = gt_depth.shape[-3:])

        mask = (gt_depth > MIN_DEPTH) & (gt_depth < MAX_DEPTH)

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        loss_gt = self.hparams.beta * self.criterion(pred_depth, gt_depth)
        loss_feature = self.feature_loss.compute()
        
        return loss_gt, loss_feature, pred_depth, gt_depth

    def training_step(
        self, batch, batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        
        loss_gt, loss_feature, pred_depth,targets = self.model_step(batch)

        loss = loss_gt + loss_feature

        self.train_loss.update(loss)
        self.train_loss_gt.update(loss_gt)
        self.train_loss_feature.update(loss_feature)
        # self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_gt", self.train_loss_gt.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_feature", self.train_loss_feature.compute(), on_step=False, on_epoch=True, prog_bar=True)
        # self.start_time = time.time()
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        self.train_loss.reset()
        self.train_loss_feature.reset()
        self.train_loss_gt.reset()
        pass

    def validation_step(self, batch, batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss_gt, loss_feature, pred_depth, targets = self.model_step(batch)
        loss = loss_gt + loss_feature
        # update and log metrics
        # print(loss)
        self.val_loss.update(loss)
        # self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.val_loss.reset()

    def test_step(self, batch, batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        
        errors = compute_errors(preds, targets)

        for i in range(7):
            self.errors[i].update(errors[i])

        self.log("test/errors", [error.compute() for error in self.errors], on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        for i in range(7):
            self.errors[i].reset()
        # pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.depth_decoder = torch.compile(self.depth_decoder)
            self.dino_encoder = torch.compile(self.dino_encoder)
            self.temporal_encoder = torch.compile(self.temporal_encoder)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

