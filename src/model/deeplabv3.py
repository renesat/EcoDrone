import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision


class TrashDetector(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=pretrained,
        )
        self.model.classifier[4] = nn.Conv2d(
            256,
            1,
            kernel_size=(1, 1),
            stride=(1, 1),
        )
        self.model.aux_classifier[4] = nn.Conv2d(
            256,
            1,
            kernel_size=(1, 1),
            stride=(1, 1),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        # print(nn.Sigmoid()(out["out"].mean(dim=1)).shape)
        loss = nn.BCELoss()(nn.Sigmoid()(out["out"]).squeeze(1), y.float())

        self.log("train_loss", loss)

        return loss

    def val_step(self, batch, batch_idx):
        print(x.shape)
        x, y = batch
        out = self.forward(x)
        loss = nn.BCELoss()(out, x)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
