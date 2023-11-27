import numpy as np
import os
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
        train_indices , test_indices = indices[split:], indices[:split]

        # Indices are saved with pickle.
        data['train_indices'] = train_indices
        data['test_indices'] = test_indices
        with open(file_path,'wb') as file:
            pickle.dump(data,file)
    return train_indices, test_indices

def result(image, mask, output, title, transparency=0.38, save_path=None):
    """ 
    Plots a 2x3 plot with comparisons of output and original image.
    """

    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(
        20, 15), gridspec_kw={'wspace': 0.025, 'hspace': 0.010})
    fig.suptitle(title, x=0.5, y=0.92, fontsize=20)

    axs[0][0].set_title("Original Mask", fontdict={'fontsize': 16})
    axs[0][0].imshow(mask, cmap='gray')
    axs[0][0].set_axis_off()

    axs[0][1].set_title("Constructed Mask", fontdict={'fontsize': 16})
    axs[0][1].imshow(output, cmap='gray')
    axs[0][1].set_axis_off()

    seg_output = mask*transparency
    seg_image = np.add(image, seg_output)/2
    axs[1][0].set_title("Original Segment", fontdict={'fontsize': 16})
    axs[1][0].imshow(seg_image, cmap='gray')
    axs[1][0].set_axis_off()

    seg_output = output*transparency
    seg_image = np.add(image, seg_output)/2
    axs[1][1].set_title("Constructed Segment", fontdict={'fontsize': 16})
    axs[1][1].imshow(seg_image, cmap='gray')
    axs[1][1].set_axis_off()

    axs[1][2].set_title("Original Image", fontdict={'fontsize': 16})
    axs[1][2].imshow(image, cmap='gray')
    axs[1][2].set_axis_off()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=90, bbox_inches='tight')

    plt.show()