import numpy as np
import torch
from denoising_diffusion_pytorch import GaussianDiffusion, Unet
from torch import nn
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
from torch.nn import functional as F

from src import ImageDataset, Trainer
from denoising_diffusion import DiffusionModel

DATASETS = ["e9_5_GLM87a_cycle1_8_8"]
DATA_ROOT = "/home/alehc/Téléchargements/mip_edof"
IMG_SIZE = 1024
MAX_IMG_SIZE = 2048

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# class DiffusionModel(nn.Module):
#     def __init__(self, model: Unet, steps: int = 5):
#         super().__init__()
#         self.steps = steps
#         self.diffusion = GaussianDiffusion(
#             model, image_size=IMG_SIZE, timesteps=self.steps, objective="pred_noise",
#         )
# 
#     def forward(self, img: torch.Tensor, return_all: bool = False) -> torch.Tensor:
#         """Computes the denosied image starting from `img`."""
#         img = self.diffusion.normalize(img)
#         imgs = []
#         for t in tqdm(reversed(range(0, self.steps)), leave=False,
#                       desc="Predicting", total=self.steps):
#             t = torch.full((img.size(0),), t, device=img.device, dtype=torch.long)
#             x_start = self.diffusion.model_predictions(img, t).pred_x_start.clamp(-1, 1)
#             img, _, _, _ = self.diffusion.
# 
#             # img = self.diffusion.model_predictions(img, t).pred_x_start.clamp(-1, 1)
# 
#             # img, _, _, _ = self.diffusion.p_mean_variance(img, t)
#             # imgs.append(img)
#         ret = torch.stack(imgs, dim=1) if return_all else img
#         return self.diffusion.unnormalize(ret).clamp(0, 1)
# 
#         # img = self.diffusion.normalize(img)
#         # imgs = [img]
#         # x_start = None
#         # for t in tqdm(reversed(range(0, self.steps)), leave=False,
#         #               desc="Predicting", total=self.steps):
#         #     self_cond = x_start if self.diffusion.self_condition else None
#         #     img, x_start = self.diffusion.p_sample(img, t, x_self_cond=self_cond)
#         #     imgs.append(img)
#         # ret = torch.stack(imgs, dim=1) if return_all else img
#         # return self.diffusion.unnormalize(ret)


def identity_closure(model: DiffusionModel, batch: dict) -> torch.Tensor:
    x = model.diffusion.normalize(batch["input_image"])
    t = torch.randint(0, model.steps, (x.size(0),), device=DEVICE, dtype=torch.long)
    pred, _, _, _ = model.diffusion.p_mean_variance(x, t)
    return F.mse_loss(pred, x)


# def diffusion_closure(model: DiffusionModel, batch: dict) -> torch.Tensor:
#     return model.diffusion(batch["output_image"].bfloat16(),
#                            noise=model.diffusion.normalize(batch["input_image"].bfloat16()))


def psnr(predicted, target, max_pixel=16800):
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
        psnr_ = psnr(16800*tensor_img_to_numpy(x), 16800*tensor_img_to_numpy(y))
        plt.subplot(3, n_show, 0*n_show + i + 1)
        plt.title(f"Image {idx.item()}")
        if i == 0:
            plt.ylabel("MIP")
        plt.imshow(tensor_img_to_numpy(x), **kw)
        plt.xlabel(f"PSNR: {psnr_:.6f}")
        plt.xticks([])
        plt.yticks([])

        # Show DT-EDF.
        psnr_ = psnr(16800*tensor_img_to_numpy(y), 16800*tensor_img_to_numpy(y))
        plt.subplot(3, n_show, 1*n_show + i + 1)
        if i == 0:
            plt.ylabel("DT-EDF")
        plt.imshow(tensor_img_to_numpy(y), **kw)
        plt.xlabel(f"PSNR: {psnr_:.6f}")
        plt.xticks([])
        plt.yticks([])

        # Get prediction and show it.
        with torch.no_grad():
            pred = model(x[None, :, :, :].bfloat16())[0]
        outside = torch.mean(((pred < 0) | (pred > 1)).float())
        if outside > 0:
            print(f"Warning! {100*outside:.1f}% of predictions lie outside [0, 1]")
        psnr_ = psnr(16800*tensor_img_to_numpy(pred), 16800*tensor_img_to_numpy(y))
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


def main():
    # Load dataset and display basic info.
    dataset = ImageDataset(DATA_ROOT, DATASETS, image_size=IMG_SIZE, normalize="minmax", normalize_output=True)
    info(ImageDataset(DATA_ROOT, DATASETS, image_size=MAX_IMG_SIZE, normalize="minmax", normalize_output=True))

    # Split dataset and make dataloaders.
    dset, dset_test = random_split(dataset, [0.8, 0.2])
    # dset = [dataset[11]]
    # dset_test = dset
    loader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=True)

    # Initialize unet and show dummy prediction.
    model = Unet(dim=32, dim_mults=(1, 2, 4, 8), flash_attn=True, channels=1,
                 attn_dim_head=8, attn_heads=2).bfloat16()
    model = DiffusionModel(model, steps=50, objective="x0").to(DEVICE).bfloat16().eval()
    n_params = sum(p.numel() for p in model.parameters())/1000/1000
    print(f"Number of parameters: {n_params:.3f}M")
    show_imgs(model, dset_test, n_show=5)

    # Start initial training: Predict identity function.
    # model = model.train()
    # trainer = Trainer(model, None, DEVICE, identity_closure)
    # trainer.train(5, loader, learning_rate=0.001)
    # model = model.eval()

    # Show image after training.
    # show_imgs(model, dset_test)

    # Start final training: Denoising.
    model = model.train()
    trainer = Trainer(model, None, DEVICE, closure=lambda model, batch: model.closure(batch))
    history = trainer.train(200, loader, weight_decay=0.0, learning_rate=1e-3,
                            scheduler="cosine", warmup=0.2)
    model = model.eval()

    # Show learning history.
    plt.plot(history, label="Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss history")
    plt.show()

    # Show image after training.
    show_imgs(model, dset_test, n_show=5)


if __name__ == "__main__":
    torch.random.manual_seed(123456789)
    main()
