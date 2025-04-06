import torch
import torch.nn as nn

class Loss_teacher(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def _normalize_depth(self, depth_tensor):
        # min-max normalization
        # per image
        B,_= depth_tensor.shape
        normalized = torch.empty_like(depth_tensor)
        for i in range(B):
            img = depth_tensor[i]
            min_val = img.min()
            max_val = img.max()
            normalized[i] = (img - min_val) / (max_val - min_val + self.eps)
        return normalized

    def _d_hat(self, d):
        # 각 배치별 계산을 위해 dim 설정
        # 여기서는 B x (HxW) 로 2차원이라고 가정
        median, _ = torch.median(d, dim=-1)  # 이러면 나오는 결과 : Bx1
        # print(median)
        t = median.unsqueeze(-1)
        t = t.expand(-1,d.shape[-1])

        # t = torch.matmul(median.unsqueeze(0), t)
        s = torch.sum(torch.abs(d - t), dim=-1) / (d.shape[-1]) + self.eps
        # s : Bx1 사이즈

        s = s.unsqueeze(-1).expand(-1,d.shape[-1])
        # print("d : ", d)
        # print("t : ", t)
        # print("s : ", s)
        # print("분자 : ", torch.sum(d - t, dim=-1))

        return (d - t) / s

    def _rho(self, pred, y):
        # print("d_hat pred: ", self._d_hat(pred))
        # print("d_hat y: ", self._d_hat(y))
        return torch.abs(self._d_hat(pred) - self._d_hat(y))

    def forward(self, pred, y, disparity=False):
        """
        :param pred: Prediction per pixel. size : BxHxW
        :param y: Ground truth. size : BxHxW
        """

        pred = pred.view(B, H * W)
        y = y.squeeze()
        y = y.view(B, H * W)

        # print(pred.shape)
        # print(y.shape)
        loss_u = 0

        if not disparity:
            ## 역수로 바꿔주기
            temp = torch.ones_like(y)
            y = temp / (y + self.eps)

        ## group_1에 대해서는 그냥 일반 ssi loss 적용해주면 ok

        y = self._normalize_depth(y)

        print("gt ( 1st element ):", y[0])
        print("pred ( 1st element ):", pred[0])
        print("y aggregation ( 1st element ):", torch.sum(y[0], dim=-1))
        print("pred aggregation ( 1st element ):", torch.sum(pred[0], dim=-1))

        loss_l = torch.sum(self._rho(pred,y), dim=-1) / pred.shape[-1]
        loss_l = loss_l.mean()

        return loss_l


"""
dummy data example

# len_data는 12 (즉, 전체 배치 12개, 1/3: labeled, 1/3: unlabeled, 1/3: cutmix)
len_data = 12
B = len_data  # 전체 배치 크기
H = 448
W = 448

# pred와 y: (B, H, W) 모양의 더미 예측 및 ground truth, 랜덤값 사용
pred = torch.randn(B, H, W)
y = torch.randn(B, H, W)

# 인코더 결과: 예시로 (B, num_patch, embedding_dim) 모양 생성
num_patch = 10
embedding_dim = 64

# Loss 모듈 인스턴스 생성
loss_module = Loss_teacher()
# loss 계산
total_loss = loss_module(pred, y, disparity=True)

print("Total loss:", total_loss.item())

"""