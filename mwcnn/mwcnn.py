from layers import DWT, IWT, default_conv, BBlock,DBlock_com1, DBlock_com, DBlock_inv1, DBlock_inv
import torch.nn as nn
import torch
# Adapted from
# https://github.com/pminhtam/MWCNN/tree/main



class MWCNN(nn.Module):
    def __init__(self, n_feats=64 ,n_colors=1, batch_normalize=True, conv=default_conv):
        super(MWCNN, self).__init__()
        n_feats = n_feats
        kernel_size = 3
        self.scale_idx = 0
        nColor = n_colors

        act = nn.ReLU(True)

        self.DWT = DWT()
        self.IWT = IWT()

        m_head = [BBlock(conv, nColor, n_feats, kernel_size, act=act)]
        d_l0 = []
        d_l0.append(DBlock_com1(conv, n_feats, n_feats, kernel_size, act=act, bn=batch_normalize))


        d_l1 = [BBlock(conv, n_feats * 4, n_feats * 2, kernel_size, act=act, bn=batch_normalize)]
        d_l1.append(DBlock_com1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=batch_normalize))

        d_l2 = []
        d_l2.append(BBlock(conv, n_feats * 8, n_feats * 4, kernel_size, act=act, bn=batch_normalize))
        d_l2.append(DBlock_com1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=batch_normalize))
        pro_l3 = []
        pro_l3.append(BBlock(conv, n_feats * 16, n_feats * 8, kernel_size, act=act, bn=batch_normalize))
        pro_l3.append(DBlock_com(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=batch_normalize))
        pro_l3.append(DBlock_inv(conv, n_feats * 8, n_feats * 8, kernel_size, act=act, bn=batch_normalize))
        pro_l3.append(BBlock(conv, n_feats * 8, n_feats * 16, kernel_size, act=act, bn=batch_normalize))

        i_l2 = [DBlock_inv1(conv, n_feats * 4, n_feats * 4, kernel_size, act=act, bn=batch_normalize)]
        i_l2.append(BBlock(conv, n_feats * 4, n_feats * 8, kernel_size, act=act, bn=batch_normalize))

        i_l1 = [DBlock_inv1(conv, n_feats * 2, n_feats * 2, kernel_size, act=act, bn=batch_normalize)]
        i_l1.append(BBlock(conv, n_feats * 2, n_feats * 4, kernel_size, act=act, bn=batch_normalize))

        i_l0 = [DBlock_inv1(conv, n_feats, n_feats, kernel_size, act=act, bn=batch_normalize)]

        m_tail = [conv(n_feats, nColor, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.d_l2 = nn.Sequential(*d_l2)
        self.d_l1 = nn.Sequential(*d_l1)
        self.d_l0 = nn.Sequential(*d_l0)
        self.pro_l3 = nn.Sequential(*pro_l3)
        self.i_l2 = nn.Sequential(*i_l2)
        self.i_l1 = nn.Sequential(*i_l1)
        self.i_l0 = nn.Sequential(*i_l0)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x0 = self.d_l0(self.head(x))
        x1 = self.d_l1(self.DWT(x0))
        x2 = self.d_l2(self.DWT(x1))
        x_ = self.IWT(self.pro_l3(self.DWT(x2))) + x2
        x_ = self.IWT(self.i_l2(x_)) + x1
        x_ = self.IWT(self.i_l1(x_)) + x0
        x = self.tail(self.i_l0(x_)) + x

        return x
    
    def predict(self, x_T: torch.Tensor) -> torch.Tensor:
        return self(x_T.to(self.device)).to(x_T.device)

def get_pretrained(path: str = "mwcnn/models/none_mwcnn_feats_32.pth"):
    model = MWCNN(n_feats=32, n_colors=1, batch_normalize=True)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval().requires_grad_(False).bfloat16()