import pytorch_lightning as pl
import torch


class TrashDetector(pl.LightningModule):
    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0",
            "deeplabv3_resnet50",
            pretrained=pretrained,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.binary_cross_entropy(out, x)

        self.log("train_loss", loss)

        return loss

    def val_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = F.binary_cross_entropy(out, x)

        self.log("train_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
