import torch
import lightning as L
import torch.nn.functional as F
from pytorch_metric_learning import losses
import kornia


class DenoiserTrainer(L.LightningModule):
    def __init__(self, Denoiser):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Denoiser.to(device)
        self.steps = 0
        self.loss_falloff, self.loss_falloff_steps = 0.1, 15000

        return

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        noisy, clean = noisy.to(self.device), clean.to(self.device)
        stage_1, stage_2 = self.model(noisy)

        alpha = self.loss_falloff ** (self.steps / self.loss_falloff_steps) if self.steps < self.loss_falloff_steps else self.loss_falloff

        loss = (kornia.losses.psnr_loss(stage_1, clean, max_val = 255.0)) * alpha
        loss += (kornia.losses.psnr_loss(stage_2, clean, max_val = 255.0)) * (1 - alpha)

        self.steps += 1
        
        # loss = (kornia.losses.psnr_loss(clean, stage_1, max_val = 255.) * alpha) + \
        #     (kornia.losses.psnr_loss(clean, stage_1, max_val = 255.) * (1 - alpha))
        
        
        self.log(
            "train/loss", 
            loss, 
            prog_bar = True, 
            on_step = True, 
            on_epoch = True
        )

        self.log(
            "train/PSNR", 
            kornia.metrics.psnr(stage_2, clean, max_val = 255.0), 
            prog_bar = True, 
            on_step = True, 
            on_epoch = True
        )

        self.log(
            "train/SSIM", 
            kornia.metrics.ssim(stage_2, clean, window_size = 11, max_val = 255.0).mean(),
            prog_bar = True, 
            on_step = True, 
            on_epoch = True
        )
        
        self.save_hyperparameters()
        return {"loss": loss}
    
    def on_train_epoch_end(self):
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'])

    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        noisy, clean = noisy.to(self.device), clean.to(self.device)
        _, stage_2 = self.model(noisy)
        
        loss = (kornia.losses.psnr_loss(stage_2, clean, max_val = 255.0))
        
        self.log(
            "val/loss", 
            loss, 
            prog_bar = True, 
            on_epoch = True
        )

        self.log(
            "val/PSNR", 
            kornia.metrics.psnr(stage_2, clean, max_val = 255.0), 
            prog_bar = True,
            on_epoch = True
        )

        self.log(
            "val/SSIM", 
            kornia.metrics.ssim(stage_2, clean, window_size = 11, max_val = 255.0).mean(),
            prog_bar = True,
            on_epoch = True
        )

        return {"loss": loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, maximize = False)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer = optimizer,
            mode = 'min',
            factor = 0.30,
            patience = 2,
            min_lr = 1e-7
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "val/loss"}]
    
    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch = self.current_epoch, metrics = metric)

    def save_Denoiser(self, path):
        torch.save(self.model, path)




