import torch


def teacher_loss_function(pred,y,disparity=True):
    """
    :param pred: Prediction per pixel. size : BxHxW
    :param y: Ground truth. size : BxHxW
    """
    B,H,W = pred.shape
    y = y.squeeze()

    #print(pred.shape)
    #print(y.shape)

    eps = 1e-8
    if not disparity:
        ## 역수로 바꿔주기
        temp = torch.ones_like(y)
        y = temp/(y+eps)

    def normalize_depth(depth_tensor):
        # depth_tensor: (1, H, W)
        # min-max normalization per image
        min_val = depth_tensor.min()
        max_val = depth_tensor.max()
        normalized = (depth_tensor - min_val) / (max_val - min_val + eps)
        return normalized

    y = normalize_depth(y)

    print("gt :",y[0])
    print("pred :",pred[0])
    print("y aggregation :" ,torch.sum(y[0][0],dim=-1))
    print("pred aggregation :" ,torch.sum(pred[0][0],dim=-1))
    #pred = torch.reshape(pred,(B,H*W))
    #y = torch.reshape(y,(B,H*W))

    pred = pred.view(B, H * W)
    y = y.view(B, H * W)
    loss_l= 0

    def d_hat(d):
        # 각 배치별 계산을 위해 dim 설정
        # 여기서는 B x (HxW) 로 2차원이라고 가정
        median, _ = torch.median(d, dim=-1) # 이러면 나오는 결과 : Bx1
        # print(median)
        t = median.unsqueeze(-1)
        t = t.expand(-1,d.shape[-1])    # 각 배치값을 확장

        #t = torch.matmul(median.unsqueeze(0),t)
        s = torch.sum(torch.abs(d-t),dim=-1)/(d.shape[-1]) + eps
        # s : Bx1 사이즈

        s = s.unsqueeze(-1).expand(-1,d.shape[-1])

        #print("d : ",d)
        #print("t : ",t)
        #print("s : ",s)
        #print("분자 : ",torch.sum(d-t,dim=-1))

        return (d-t)/s

    def rho(pred,y):
        #print("d_hat pred: ",d_hat(pred))
        #print("d_hat y: ",d_hat(y))
        return torch.abs(d_hat(pred) - d_hat(y))

    loss_l = torch.sum(rho(pred,y),dim=-1) / pred.shape[-1]
    #print(loss_l)

    return loss_l.mean()