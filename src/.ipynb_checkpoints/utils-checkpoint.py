import numpy as np
import os
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import torch

'''
Nothing needs to be changed in this file unless you want to play around.
'''


def calculate_fft(img):
    fft_im = torch.view_as_real(torch.torch.fft.fft2(img))
    fft_amp = fft_im[:,:,:,0]**2 + fft_im[:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,1], fft_im[:,:,:,0])
    return fft_amp, fft_pha


class FFTloss(torch.nn.Module):
    def __init__(self, loss_f = torch.nn.L1Loss):
        super(FFTloss, self).__init__()
        self.criterion = loss_f()

    def forward(self, pred, target):
        target = target.to(torch.float32)
        # Apply hann window first
        han_window = torch.sqrt(torch.outer(torch.hamming_window(32), torch.hamming_window(32)))
        han_window = han_window[None, None, :, :]
        han_pred = F.conv2d(pred,han_window, padding=1)
        han_target = F.conv2d(target,han_window, padding=1)
        # apply fft
        pred_amp, pred_pha = calculate_fft(han_pred)
        target_amp, target_pha = calculate_fft(han_target)
        loss = 0.5*self.criterion(pred_amp, target_amp) + 0.5*self.criterion(pred_pha, target_pha)
        return loss
    

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
            return data['train_indices'], data['test_indices']
        
    else:
        # File not found or fresh copy is required.
        indices = list(range(length))
        np.random.shuffle(indices)
        split = int(np.floor(data_split * length))
        train_indices ,validation_indices, test_indices = indices[2*split:],indices[split:2*split] ,indices[:split]

        # Indices are saved with pickle.
        data['train_indices'] = train_indices
        data['test_indices'] = test_indices
        data['validation_indices'] = test_indices
        with open(file_path,'wb') as file:
            pickle.dump(data,file)
    return train_indices, validation_indices,test_indices

def plot_result(results, title, save_path=None):
    """ 
    Plots a len(results)x3 plot with comparisons of output and original image.
    Results is a list of dicts with keys: 'MIP', 'pred', 'EDOF', 'original_score', 'improved_score
    """

    fig, axs = plt.subplots(len(results), 3, sharex=True, sharey=True, figsize=(
        20, 15), 
        gridspec_kw={'wspace': 0.025, 'hspace': 0.10}
    )
    fig.suptitle(title, x=0.5, y=0.92, fontsize=20)
    for i in range(len(results)):
        axs[i][0].set_title(f"Original MIP PSNR: {results[i]['original_score']}", fontdict={'fontsize': 16})
        axs[i][0].imshow(results[i]["MIP"], cmap='gray')
        axs[i][0].set_axis_off()

        axs[i][1].set_title(f"Denoised MIP, PSNR: {results[i]['improved_score']}", fontdict={'fontsize': 16})
        axs[i][1].imshow(results[i]["pred"], cmap='gray')
        axs[i][1].set_axis_off()

        axs[i][2].set_title("Original EDOF", fontdict={'fontsize': 16})
        axs[i][2].imshow(results[i]["EDOF"], cmap='gray')
        axs[i][2].set_axis_off()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches='tight')

    plt.show()