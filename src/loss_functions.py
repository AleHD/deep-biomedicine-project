import torch
import torchvision
import torch.nn.functional as F

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
    

def calculate_fft(img):
    fft_im = torch.view_as_real(torch.torch.fft.fft2(img, norm="backward"))
    fft_amp = fft_im[:,:,:,0]**2 + fft_im[:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,1], fft_im[:,:,:,0])
    return fft_amp, fft_pha


class FFTloss(torch.nn.Module):
    def __init__(self, device, loss_f = torch.nn.L1Loss, han_window=False):
        super(FFTloss, self).__init__()
        self.criterion = loss_f()
        self.han_window = han_window
        self.device = device

    def forward(self, pred, target):
        target = target.to(torch.float32)
        # Apply hann window first
        if self.han_window:
            han_window = torch.sqrt(torch.outer(torch.hamming_window(32), torch.hamming_window(32)))
            han_window = han_window[None, None, :, :].to(self.device)
            han_pred = F.conv2d(pred,han_window, padding=1)
            han_target = F.conv2d(target,han_window, padding=1)
            # apply fft
            pred_amp, pred_pha = calculate_fft(han_pred)
            target_amp, target_pha = calculate_fft(han_target)
        else:
            pred_amp, pred_pha = calculate_fft(pred)
            target_amp, target_pha = calculate_fft(target)
        loss = self.criterion(pred_amp, target_amp) #+ 0.5*self.criterion(pred_pha, target_pha)
        return loss