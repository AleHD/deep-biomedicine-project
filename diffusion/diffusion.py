import sys
sys.path.append(".")

import random

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt

from src.dataset import get_splits
from src.utils import plot_loss
from denoising_diffusion import DiffusionModel
from diffusion_trainer import DiffusionTrainer
from unet_model import Unet


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_model() -> DiffusionModel:
    base = Unet(dim=32, dim_mults=(1, 2, 4, 8), flash_attn=True, channels=1,
                attn_dim_head=8, attn_heads=2)
    return DiffusionModel(base, steps=10, objective="x0")


def get_pretrained(path: str = "diffusion/model.pt") -> DiffusionModel:
    model = get_model()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval().requires_grad_(False).bfloat16()


def main():
    # Load dataset and make dataloaders.
    dset, dset_test = get_splits("data", normalize="minmax")
    dset, dset_val = random_split(dset, [0.8, 0.2])
    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True)
    loader_val = torch.utils.data.DataLoader(dset_val, batch_size=1, shuffle=False)
    print("Dataset length:", len(dset))

    # Initialize model.
    model = get_model().to(DEVICE).bfloat16().eval()
    n_params = sum(map(torch.numel, model.parameters()))/1000/1000
    print(f"Number of parameters: {n_params:.3f}M")

    # Start training.
    epochs = 3
    model = model.train()
    trainer = DiffusionTrainer(model, learning_rate=1e-3, weight_decay=0.0)
    train_loss, val_loss = trainer.train(epochs, loader, loader_val,
                                         scheduler="cosine", warmup=0.2)
    model = model.eval()

    # Save model.
    torch.save(model.state_dict(), "diffusion/model.pt")

    # Show learning history.
    plot_loss(epochs, train_loss, val_loss)
    plt.savefig("diffusion/learning.png")


if __name__ == "__main__":
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    main()
