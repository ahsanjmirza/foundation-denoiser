import torch
from trainers.FoundationTrainer import FoundationTrainer
from models import Foundation, SLearner
import os

FOUNDATION_CKPT_PATH = os.path.join(
    "./foundation_ckpt/lightning_logs/version_0/checkpoints", 
    os.listdir("./foundation_ckpt/lightning_logs/version_0/checkpoints")[0]
)
FOUNDATION_MODEL_PATH = "./foundation_ckpt/foundation.pth"
OUT_CHANNELS = 32

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_Foundation():
    backbone = Foundation.FoundationISP(OUT_CHANNELS)
    learner = SLearner.SSLearner(OUT_CHANNELS + 1, 3)

    model = FoundationTrainer.load_from_checkpoint(
        FOUNDATION_CKPT_PATH,
        Learner = learner,
        Backbone = backbone
    ).to(device)

    return model.bb


def save_Foundation():
    model = torch.jit.trace(
        load_Foundation(), 
        torch.tensor(torch.rand(size = (1, 3, 256, 256), dtype = torch.float32)).to(device)
    )
    torch.jit.save(model, FOUNDATION_MODEL_PATH)
    
save_Foundation()


