import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt

from src.dataset import get_splits
from .denoising_diffusion import DiffusionModel
from .diffusion_trainer import DiffusionTrainer
from .unet import Unet


DATASETS = ["e9_5_GLM87a_cycle1_8_8"]
DATA_ROOT = "/home/alehc/Téléchargements/mip_edof"
IMG_SIZE = 1024
MAX_IMG_SIZE = 2048

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def psnr(predicted, target, max_pixel=1):
    mse = np.mean((predicted - target) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr 


def show_imgs(model: DiffusionModel, dataset: Dataset, n_show: int = 4) -> None:
    """Shows images, denosied images and predictions."""

    def tensor_img_to_numpy(img: torch.Tensor) -> np.ndarray:
        """Transpose a (C, H, W) to (H, W, C)"""
        return img.permute(1, 2, 0).detach().cpu().double().numpy()

    kw = {"cmap": "gray", "vmin": 0, "vmax": 1}
    batch = next(iter(DataLoader(dataset, shuffle=True, batch_size=n_show)))
    for i in range(n_show):
        # Extract data.
        idx = batch["index"][i]
        x = batch["input_image"][i].to(DEVICE)
        y = batch["output_image"][i].to(DEVICE)

        # Show MIP.
        psnr_ = psnr(tensor_img_to_numpy(x), tensor_img_to_numpy(y))
        plt.subplot(3, n_show, 0*n_show + i + 1)
        plt.title(f"Image {idx.item()}")
        if i == 0:
            plt.ylabel("MIP")
        plt.imshow(tensor_img_to_numpy(x), **kw)
        plt.xlabel(f"PSNR: {psnr_:.6f}")
        plt.xticks([])
        plt.yticks([])

        # Show DT-EDF.
        psnr_ = psnr(tensor_img_to_numpy(y), tensor_img_to_numpy(y))
        plt.subplot(3, n_show, 1*n_show + i + 1)
        if i == 0:
            plt.ylabel("DT-EDF")
        plt.imshow(tensor_img_to_numpy(y), **kw)
        plt.xlabel(f"PSNR: {psnr_:.6f}")
        plt.xticks([])
        plt.yticks([])

        # Get prediction and show it.
        with torch.no_grad():
            pred = model.predict(x[None, :, :, :].bfloat16())[0]
        outside = torch.mean(((pred < 0) | (pred > 1)).float())
        if outside > 0:
            print(f"Warning! {100*outside:.1f}% of predictions lie outside [0, 16800]")
        psnr_ = psnr(tensor_img_to_numpy(pred), tensor_img_to_numpy(y))
        plt.subplot(3, n_show, 2*n_show + i + 1)
        if i == 0:
            plt.ylabel("Prediction")
        plt.imshow(tensor_img_to_numpy(pred), **kw)
        plt.xlabel(f"PSNR: {psnr_:.6f}")
        plt.xticks([])
        plt.yticks([])
    plt.suptitle("Prediction plot")
    plt.tight_layout()
    plt.show()


def info(dataset: Dataset) -> None:
    """Just prints some general dataset info."""

    print("Dataset info:")
    min_val = float("inf")
    max_val = -float("inf")
    print("Length:", len(dataset))
    for sample in tqdm(dataset, desc="Calculating min-max values"):
        min_val = min(min_val, sample["input_image"].min().item(),
                      sample["output_image"].min().item())
        max_val = max(max_val, sample["input_image"].max().item(),
                      sample["output_image"].max().item())
    print("Min:", min_val)
    print("Max:", max_val)


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
    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True)
    print("Dataset length:", len(dset))

    # Initialize model.
    model = get_model().to(DEVICE).bfloat16().eval()
    n_params = sum(map(torch.numel, model.parameters()))/1000/1000
    print(f"Number of parameters: {n_params:.3f}M")

    # Start training.
    model = model.train()
    trainer = DiffusionTrainer(model, learning_rate=1e-3, weight_decay=0.0)
    history = trainer.train(50, loader, scheduler="cosine", warmup=0.2)
    model = model.eval()

    # Save model.
    torch.save(model.state_dict(), "diffusion/model.pt")

    # Show learning history.
    plt.plot(history, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss history")
    plt.show()
    plt.savefig("diffusion/learning.png")


if __name__ == "__main__":
    torch.random.manual_seed(123456789)
    main()
