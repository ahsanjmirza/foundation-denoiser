import torch
import lightning as L
import torch.nn.functional as F
from pytorch_metric_learning import losses
import kornia


class FoundationTrainer(L.LightningModule):
    def __init__(self, Learner, Backbone):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bb = Backbone.to(device)
        self.ssl = Learner.to(device)
        
    def forward(self, x):
        return self.bb(x)

    def training_step(self, batch, batch_idx):
        distorted, orginal, mode = batch
        distorted, orginal, mode = distorted.to(self.device), orginal.to(self.device), mode.to(self.device)
        undistored = self.ssl(self.bb(distorted / 255.), mode) * 255.
        
        loss = kornia.losses.psnr_loss(orginal, undistored, max_val = 255.)
        
        self.log("train/PSNRLoss", loss, prog_bar = True, on_step = True, on_epoch = True)
        self.save_hyperparameters()
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'])

    def validation_step(self, batch, batch_idx):
        distorted, orginal, mode = batch
        distorted, orginal, mode = distorted.to(self.device), orginal.to(self.device), mode.to(self.device)
        undistored = self.ssl(self.bb(distorted / 255.), mode) * 255.
        
        loss = kornia.losses.psnr_loss(orginal, undistored, max_val = 255.)
        
        self.log("val/PSNRLoss", loss, prog_bar = True)
        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, maximize = False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            mode = 'min',
            factor = 0.30,
            patience = 2,
            min_lr = 1e-8
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val/PSNRLoss"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch = self.current_epoch, metrics = metric)

    def save_Backbone(self, path):
        torch.save(self.bb, path)




