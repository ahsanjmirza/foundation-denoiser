from trainers.DenoiserTrainer import DenoiserTrainer
from models import Denoiser
from dataloader import DenoiseDL
from torch.utils.data import DataLoader
import lightning as L

TRAIN_DATASET_DIR = './dataset/train'
VAL_DATASET_DIR = './dataset/val'
IMAGE_SHAPE = (128, 128)
OUT_CHANNELS = 32
BATCH_SIZE = 4
MIN_EPOCHS = 1
MAX_EPOCHS = 50
SAVE_DIR = './denoiser_ckpt'
CONTINUE_TRAINING = ""
NUM_WORKERS = 4


if __name__ == "__main__":
    denoiser = Denoiser.Model(OUT_CHANNELS)

    train_loader = DataLoader(   
        DenoiseDL(TRAIN_DATASET_DIR, IMAGE_SHAPE[0]),
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = NUM_WORKERS, 
        persistent_workers = True
    )

    val_loader = DataLoader(
        DenoiseDL(VAL_DATASET_DIR, IMAGE_SHAPE[0]),
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        persistent_workers = True
    )
 
    trainer = L.Trainer(
        min_epochs = MIN_EPOCHS,
        max_epochs = MAX_EPOCHS,
        default_root_dir = SAVE_DIR,
        log_every_n_steps = 20
    )

    model = DenoiserTrainer(denoiser)

    if CONTINUE_TRAINING != "":
        trainer.fit(
            model = model, 
            train_dataloaders = train_loader, 
            val_dataloaders = val_loader,
            ckpt_path = CONTINUE_TRAINING 
        )
    else: 
        trainer.fit(
            model = model, 
            train_dataloaders = train_loader, 
            val_dataloaders = val_loader
        )
