import copy
from collections import defaultdict
from typing import Callable, Optional

import numpy as np
import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset import get_splits, ImageDataset
from diffusion import get_pretrained

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model: Callable[[torch.Tensor], torch.Tensor], dataset: ImageDataset,
             im_size: int = 1024, max_val: int = 16800, batch_size: int = 8,
             dset_config: Optional[dict] = None) -> dict[str, float]:
    """
    Returns a dict with evaluation metrics of the `model` in the `dataset`.
    The `model` will need to be a function that takes a batch of images
    (i.e. size [batch_size, 1, 1024, 1024]) and returns the unnoised image (tensor of the
    same size).

    The `dset_config` should potentially be a dict with a "normalize" key and
    "normalize_output" key, that will indicate the dataset configuration
    used by `model` during training.
    """

    def apply_preprocessing(x: torch.Tensor,
                            pred: Optional[torch.Tensor] = None
                            ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns x processed according to `dset_config["normalize"]`.
        If pred is not None and `dset_config["normalize_output"]` is true,
        then it also returns `pred` with an inverse normalization operation.
        """

        normalize = dset_config.get("normalize", "standard")
        normalize_output = dset_config.get("normalize_output", False)

        if normalize == "standard":
            mean = torch.mean(x.reshape(batch_size, -1), dim=1).view(x.size(0), 1, 1, 1)
            std = torch.std(x.reshape(batch_size, -1), dim=1).view(x.size(0), 1, 1, 1)
            x = (x - mean)/std
            if pred is not None and normalize_output:
                pred = pred*std + mean
            return x, pred
        if normalize == "minmax":
            mi = torch.min(x.reshape(batch_size, -1), dim=1).values.view(x.size(0), 1, 1, 1)
            ma = torch.max(x.reshape(batch_size, -1), dim=1).values.view(x.size(0), 1, 1, 1)
            x = (x - mi)/(ma - mi)
            if pred is not None and normalize_output:
                pred = pred*(ma - mi) + mi
            return x, pred
        return x, pred

    if dset_config is None:
        dset_config = {}

    dataset = copy.deepcopy(dataset)
    dataset.default_transformation = v2.Compose([v2.ToTensor(), v2.ToDtype(torch.float32)])
    loader = DataLoader(dataset, batch_size=batch_size)
    metrics = defaultdict(list)
    for sample in tqdm(loader, desc="Evaluating"):
        x = sample["input_image"]
        y = sample["output_image"]
        assert x.size() == y.size()
        assert x.size(2) % im_size == 0
        assert x.size(3) % im_size == 0
        for i in range(0, x.size(2), im_size):
            for j in range(0, x.size(3), im_size):
                x_cropped = x[:, :, i : i+im_size, j : j+im_size]
                y_cropped = y[:, :, i : i+im_size, j : j+im_size]
                x_cropped_processed, _ = apply_preprocessing(x_cropped)

                pred_processed = model(x_cropped_processed.to(DEVICE)).cpu()
                _, pred = apply_preprocessing(x_cropped, pred=pred_processed)

                dif = (pred - y_cropped).view(batch_size, -1)
                mse = torch.mean(dif**2, dim=1)
                mae = torch.mean(torch.abs(dif), dim=1)
                psnr = torch.where(mse == 0, 100, 20*torch.log10(max_val/torch.sqrt(mse)))

                metrics["mse"] += mse.tolist()
                metrics["mae"] += mae.tolist()
                metrics["psnr"] += psnr.tolist()

    return {metric: np.mean(vals) for metric, vals in metrics.items()}


def main():
    # Evaluate baseline.
    print("Evaluating identity")
    _, dset_test = get_splits("data")
    results = evaluate(lambda x: x, dset_test, dset_config={"normalize": "none"})
    print("Identity")
    print(results)

    # Evaluate diffusion.
    print("Evaluating diffusion")
    model = get_pretrained().to(DEVICE)
    results = evaluate(model.predict, dset_test, dset_config={"normalize": "minmax",
                                                              "normalize_output": True})
    print("Diffusion")
    print(results)



if __name__ == "__main__":
    main()
