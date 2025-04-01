import torch


def student_loss_function(pred,y,disparity=True):
    """
    :param pred: Prediction per pixel. size : BxHxW
    :param y: Ground truth. size : BxHxW
    """
    B,H,W = pred.shape
    y = y.squeeze()

    eps = 1e-8
    if not disparity:
        ## 역수로 바꿔주기
        temp = torch.ones_like(y)
        y = temp/(y+eps)

    def normalize_depth(depth_tensor):
        min_val = depth_tensor.min()
        max_val = depth_tensor.max()
        normalized = (depth_tensor - min_val) / (max_val - min_val + eps)
        return normalized

    y = normalize_depth(y)

    print("gt :",y[0])
    print("pred :",pred[0])
    print("y aggregation :" ,torch.sum(y[0][0],dim=-1))
    print("pred aggregation :" ,torch.sum(pred[0][0],dim=-1))

    pred = pred.view(B, H * W)
    y = y.view(B, H * W)

    def d_hat(d):
        median, _ = torch.median(d, dim=-1)
        t = median.unsqueeze(-1)
        t = t.expand(-1,d.shape[-1])
        s = torch.sum(torch.abs(d-t),dim=-1)/(d.shape[-1]) + eps
        s = s.unsqueeze(-1).expand(-1,d.shape[-1])
        return (d-t)/s

    def rho(pred,y):
        return torch.abs(d_hat(pred) - d_hat(y))

    loss_l = torch.sum(rho(pred,y),dim=-1) / pred.shape[-1]

    return loss_l.mean()