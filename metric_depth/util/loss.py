import torch
from torch import nn

class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, eps=1e-6): # 設置一個很小的值
        super().__init__()
        self.lambd = lambd
        self.eps = eps 

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        
        # 🌟 關鍵修正：使用 clamp(min=self.eps) 確保 pred 不為零
        pred_clamped = pred[valid_mask].clamp(min=self.eps)
        target_clamped = target[valid_mask].clamp(min=self.eps)
        
        diff_log = torch.log(target_clamped) - torch.log(pred_clamped)
        
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss