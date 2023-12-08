import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import torch

'''
Nothing needs to be changed in this file unless you want to play around.
'''
    
def get_indices(length, dataset_path, data_split, new=False):
    """ 
    Gets the Training & Testing data indices
    """

    # Pickle file location of the indices.
    file_path = os.path.join(dataset_path,'split_indicess.p')
    data = dict()
    
    if os.path.isfile(file_path) and not new:
        # File found.
        with open(file_path,'rb') as file :
            data = pickle.load(file)
            return data['train_indices'],data['validation_indices']
        
    else:
        # File not found or fresh copy is required.
        indices = list(range(length))
        np.random.shuffle(indices)
        split = int(np.floor(data_split * length))
        validation_indices, train_indices = indices[:split], indices[split:]

        # Indices are saved with pickle.
        data['train_indices'] = train_indices
        data['validation_indices'] = validation_indices

        with open(file_path,'wb') as file:
            pickle.dump(data,file)

    return train_indices, validation_indices


def psnr(predicted, target, max_pixel=16384):
    """
    Predicted: the prediction from the model.
    Target: the groud truth.
    """
    mse = np.mean((predicted - target) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                # Therefore PSNR have no importance. 
        return 100
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) 
    return psnr


def plot_loss(num_epochs,train_losses, title='Training Loss Curve', label='Training Loss'):
    plt.plot(range(1, num_epochs + 1), train_losses, label=label)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_result(results, title, save_path=None):
    """ 
    Plots a len(results)x3 plot with comparisons of output and original image.
    Results is a list of dicts with keys: 'MIP', 'EDOF', and the rest of the keys
    should have the name of the model that generated it.
    Example: {'MIP': (mip image), 'EDOF': (edof image), 'GAN': (gan image),
              'diffusion': (diffusion image)}
    """

    n_models = len(results[0])
    fig, axs = plt.subplots(
        len(results), n_models, sharex=True, sharey=True, figsize=(20, 15), 
        gridspec_kw={'wspace': 0.025, 'hspace': 0.10}
    )
    fig.suptitle(title, x=0.5, y=0.92, fontsize=20)
    for i in range(len(results)):
        pnsr_ = psnr(results[i]["MIP"], results[i]["EDOF"])
        axs[i][0].set_title(f"Original MIP, PSNR: {pnsr_:.3f}", fontdict={'fontsize': 16})
        axs[i][0].imshow(results[i]["MIP"], cmap='gray')
        axs[i][0].set_axis_off()

        axs[i][1].set_title("Original EDOF", fontdict={'fontsize': 16})
        axs[i][1].imshow(results[i]["EDOF"], cmap='gray')
        axs[i][1].set_axis_off()

        for j, key in enumerate(filter(lambda name: name not in {"MIP", "EDOF"}, results[i])):
            pnsr_ = psnr(results[i][key], results[i]["EDOF"])
            axs[i][j + 2].set_title(f"Denoised with {key}, PSNR: {psnr_:.3f}", fontdict={'fontsize': 16})
            axs[i][j + 2].imshow(results[i][key], cmap='gray')
            axs[i][j + 2].set_axis_off()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches='tight')

    plt.show()
