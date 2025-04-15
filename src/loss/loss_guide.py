import torch
import torch.nn as nn

class Loss_guide(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _normalize_depth(self, depth_tensor):
        B,_= depth_tensor.shape
        normalized = torch.empty_like(depth_tensor)
        for i in range(B):
            img = depth_tensor[i]
            min_val = img.min()
            max_val = img.max()
            normalized[i] = (img - min_val) / (max_val - min_val + self.eps)
        return normalized

    def _d_hat(self, d):
        median, _ = torch.median(d, dim=-1)
        t = median.unsqueeze(-1)
        t = t.expand(-1,d.shape[-1])
        s = torch.sum(torch.abs(d - t), dim=-1) / (d.shape[-1]) + self.eps
        s = s.unsqueeze(-1).expand(-1,d.shape[-1])
        return (d - t) / s

    def _rho(self, pred, y):
        return torch.abs(self._d_hat(pred) - self._d_hat(y))

    def forward(self, pred, y, disparity=True):
        B, H, W = pred.shape

        pred = pred.view(B, H * W)
        y = y.squeeze()
        y = y.view(B, H * W)

        if not disparity:
            temp = torch.ones_like(y)
            y = temp / (y + self.eps)
        y = self._normalize_depth(y)

        print("gt ( 1st element ):", y[0])
        print("pred ( 1st element ):", pred[0])
        print("y aggregation ( 1st element ):", torch.sum(y[0], dim=-1))
        print("pred aggregation ( 1st element ):", torch.sum(pred[0], dim=-1))

        loss_l = torch.sum(self._rho(pred,y), dim=-1) / pred.shape[-1]
        loss_l = loss_l.mean()

        return loss_l

