from trainers.FoundationTrainer import FoundationTrainer
from models import Foundation, SLearner
from dataloader import DistortDL
from torch.utils.data import DataLoader
import lightning as L

TRAIN_DATASET_DIR = './dataset/train'
VAL_DATASET_DIR = './dataset/val'
IMAGE_SHAPE = (128, 128)
BATCH_SIZE = 4
LEARNER_IN_CHANNELS = 32
LEARNER_OUT_CHANNELS = 3
MIN_EPOCHS = 1
MAX_EPOCHS = 35
SAVE_DIR = './foundation_ckpt'
CONTINUE_TRAINING = ""
NUM_WORKERS = 4

if __name__ == "__main__":

    backbone = Foundation.FoundationISP(LEARNER_IN_CHANNELS)
    learner = SLearner.SSLearner(LEARNER_IN_CHANNELS + 1, LEARNER_OUT_CHANNELS)

    train_loader = DataLoader(   
        DistortDL(TRAIN_DATASET_DIR, IMAGE_SHAPE[0]),
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = NUM_WORKERS,
        persistent_workers = True
    )

    val_loader = DataLoader(
        DistortDL(VAL_DATASET_DIR, IMAGE_SHAPE[0]),
        batch_size = 8,
        num_workers = NUM_WORKERS,
        persistent_workers = True
    )

    trainer = L.Trainer(
        min_epochs = MIN_EPOCHS,
        max_epochs = MAX_EPOCHS,
        default_root_dir = SAVE_DIR,
        log_every_n_steps = 20
    )

    model = FoundationTrainer(learner, backbone)

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

