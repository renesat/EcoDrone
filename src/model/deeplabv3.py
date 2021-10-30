import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TVF
from torchvision.utils import draw_segmentation_masks


class TrashDetector(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(  # torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=pretrained,
        )
        self.model.classifier[4] = nn.Conv2d(
            256,
            1,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        self.model.aux_classifier[4] = nn.Conv2d(
            10,
            1,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = nn.Sigmoid()(out["out"]).squeeze(1)
        loss = nn.BCELoss()(out.float(), y.float())

        self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = nn.Sigmoid()(out["out"]).squeeze(1)
        loss = nn.BCELoss()(out.float(), y.float())

        out_result = (out > 0.5).bool()
        iou = (out_result & y.bool()).sum(dim=(1, 2)) / (
            (out_result | y.bool()).sum(dim=(1, 2)) + 1e-8
        )
        dsc = (
            2
            * (out_result & y.bool()).sum(dim=(1, 2))
            / (out_result.sum(dim=(1, 2)) + y.sum(dim=(1, 2)))
        )

        self.log("val/loss", loss)

        if batch_idx == 0:
            img = (
                x * torch.Tensor([0.229, 0.224, 0.225]).to(x.device).resize(1, 3, 1, 1)
                + torch.Tensor(
                    [
                        0.485,
                        0.456,
                        0.406,
                    ]
                )
                .to(x.device)
                .resize(1, 3, 1, 1)
            )
            img[img > 1] = 1
            img[img < 0] = 0
            self.logger.experiment.add_image(
                "val/img1",
                draw_segmentation_masks(
                    (img[0] * 255).type(torch.ByteTensor),
                    out_result[0],
                    alpha=0.8,
                ),
                self.current_epoch,
            )
            self.logger.experiment.add_image(
                "val/img2",
                draw_segmentation_masks(
                    (img[1] * 255).type(torch.ByteTensor),
                    out_result[1],
                    alpha=0.8,
                ),
                self.current_epoch,
            )

        return {
            "loss": loss,
            "iou": iou,
            "dsc": dsc,
        }

    def validation_epoch_end(self, epoches_output):
        mIoU = []
        mDSC = []
        for item in epoches_output:
            mIoU.extend(list(item["iou"].detach().cpu().numpy()))
            mDSC.extend(list(item["dsc"].detach().cpu().numpy()))
        mIoU = np.mean(mIoU)
        mDSC = np.mean(mDSC)
        self.log("val/IoU", mIoU)
        self.log("val/DSC", mDSC)

        print(f"mIoU = {mIoU}")
        print(f"mDSC = {mDSC}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
