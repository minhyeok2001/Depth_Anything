import torch

def scale_shift_correction(pred_batch,y_batch):
    temp = []
    for pred,y in zip(pred_batch,y_batch):
        flat_pred = pred.flatten()
        flat_y = y.flatten()

        pred_hat = torch.stack([flat_pred,torch.ones_like(flat_pred)],dim=1)
        element_1 = pred_hat.T @ pred_hat
        element_2 = pred_hat.T @ flat_y
        h_opt = torch.inverse(element_1) @ element_2

        s = h_opt[0]
        t = h_opt[1]
        temp.append(pred*s+t)

    return torch.stack(temp,dim=0)


def compute_abs_rel(pred, gt, eps=1e-8):
    pred = pred.float()  #
    gt = gt.float()
    gt = gt.squeeze(1)   # B H W
    mask = gt > eps
    abs_rel = torch.abs(pred[mask] - gt[mask]) / (gt[mask] + eps)
    return abs_rel.mean()

def compute_delta1(pred, gt, threshold=1.25, eps=1e-8):
    pred = pred.float()
    gt = gt.float()  # 불필요 1 차원 제거
    if gt.dim() == 4 and gt.size(1) == 1:
        gt = gt.squeeze(1)

    mask = gt > eps
    # mask를 사용해 유효 픽셀의 pred, gt 값을 선택하면 1D 벡터가 됨
    ratio = torch.max(pred[mask] / (gt[mask] + eps), gt[mask] / (pred[mask] + eps))
    delta1 = (ratio < threshold).float().mean()
    return delta1
