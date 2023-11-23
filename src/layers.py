import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[: ,:, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
    

class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        #print([in_batch, in_channel, in_height, in_width])
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (r ** 2)), r * in_height, r * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        

        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float()

        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

        return h

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
