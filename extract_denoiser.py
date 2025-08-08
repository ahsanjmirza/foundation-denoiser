import torch
from trainers.DenoiserTrainer import DenoiserTrainer
from models import Denoiser
import os

DENOISER_CKPT_PATH = os.path.join(
    "./denoiser_ckpt/lightning_logs/version_0/checkpoints", 
    os.listdir("./denoiser_ckpt/lightning_logs/version_0/checkpoints")[0]
)
DENOISER_MODEL_PATH = "./denoiser_ckpt/denoiser.pth"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

OUT_CHANNELS = 32

def load_Denoiser():
    denoiser = Denoiser.Model(OUT_CHANNELS)

    model = DenoiserTrainer.load_from_checkpoint(
        DENOISER_CKPT_PATH, 
        Denoiser = denoiser
    ).to(device)

    return model.model


def save_Denoiser():
    model = torch.jit.trace(
        load_Denoiser(), 
        torch.tensor(torch.rand(size = (1, 3, 128, 128), dtype = torch.float32)).to(device)
    )
    torch.jit.save(model, DENOISER_MODEL_PATH)

save_Denoiser()


