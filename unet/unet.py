import sys
sys.path.insert(0,".")

from model import UNet
from unet_trainer import *
from src.dataset import get_splits , ImageDataset 
from src.utils import get_indices
from torch.utils.data import Dataset, DataLoader ,SubsetRandomSampler

import torch
import gc
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



def psnr(predicted, target, max_pixel=1):
    mse = np.mean((predicted - target) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr 

def show_imgs(Unet_trainer, dataset: Dataset, n_show: int = 4) -> None:
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
            pred = Unet_trainer.predict(x[None, :, :, :].bfloat16())[0]
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



def main():

    # Dataset part used for testing
    VALIDATION_SPLIT = 0.15
    # Batch size for training. Limited by GPU memory
    BATCH_SIZE = 1
    # Full Dataset path
    TEST_DATASETS = ['val']
    TRAIN_DATASETS = ['train']

    ROOTDIR = './data/'

    
    """ # Load dataset and make dataloaders.
    dset, dset_val = get_splits("data", normalize="none")
    trainloader = torch.utils.data.DataLoader(dset, batch_size=2, shuffle=True)
    validationloader = torch.utils.data.DataLoader(dset_val, batch_size=2, shuffle=True)
    #print("Dataset length:", len(dset)) """

    # Load dataset and make dataloaders.
    train_image_dataset = ImageDataset(ROOTDIR, TRAIN_DATASETS, normalize="none")
    
    train_indices, validation_indices = get_indices(len(train_image_dataset), train_image_dataset.root_dir, VALIDATION_SPLIT, new=True)
    train_sampler, validation_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(validation_indices)
    
    trainloader = torch.utils.data.DataLoader(train_image_dataset, BATCH_SIZE, sampler=train_sampler)
    
    validationloader = torch.utils.data.DataLoader(train_image_dataset, BATCH_SIZE, sampler=validation_sampler)

    test_image_dataset = ImageDataset(ROOTDIR, TEST_DATASETS, normalize="none")


    #Empty GPU cache
    gc.collect()
    torch.cuda.empty_cache()

    # Training Epochs
    EPOCHS = 2
    # Filters used in UNet Model
    filter_num = [16,32,64,128,256]

    MODEL_NAME = f"models/Seif-UNet-{filter_num}.pt"

    unet_model = UNet(filter_num).to(DEVICE)

    # Start training.
    unet_trainer = Unet_trainer(unet_model,device=DEVICE, learning_rate=0.002)
    
    
    history = unet_trainer.train(EPOCHS,trainloader,validationloader)
    print(history)
    print("training done")

    # Save model.
    torch.save(unet_model.state_dict(), MODEL_NAME)

    # Show learning history.
    plt.plot(history[0], label="Training loss")
    plt.plot(history[1], label="Validation loss")
    
 
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.title("Loss history")
    plt.show()

    # Show image after training.
    #show_imgs(unet_trainer, test_image_dataset, n_show=5)

if __name__ == "__main__":
    torch.random.manual_seed(123456789)
    main()    




