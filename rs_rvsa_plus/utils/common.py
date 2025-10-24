
import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def select_gpus(gpus: str|None):
    if gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus

def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0
