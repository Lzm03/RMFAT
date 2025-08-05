import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss
    
class dwtLoss(nn.Module):

    def __init__(self, loss_weight=1.0, wkd_level=4, wkd_basis='haar', reduction='mean'):
        super(dwtLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.xfm = DWTForward(J=wkd_level, mode='zero', wave=wkd_basis)

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        _, _, H, W = pred.shape
        pred_l, pred_h = self.xfm(pred)
        target_l, target_h = self.xfm(target)
        loss = 0.0

        for index in range(len(pred_h)):
            loss += torch.nn.functional.l1_loss(target_h[index], pred_h[index], reduction=self.reduction)

        loss += torch.nn.functional.l1_loss(target_l[index], pred_l[index], reduction=self.reduction)

        return self.loss_weight * loss