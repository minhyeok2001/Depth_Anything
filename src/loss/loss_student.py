import torch
import torch.nn as nn

class Loss_student(nn.Module):
    def __init__(self, eps=1e-8,threshold=0.85):
        super().__init__()
        self.eps = eps
        self.threshold = threshold

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
        t = t.expand_as(d)  # 각 배치값을 확장

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

    def _cosine_loss(self, encoder_result, frozen_encoder_result):
        """
        결국 인코더 결과도 B x num_patch x embedding_dim 꼴이어야 함.
        """
        _, num_patch, embedding_dim = encoder_result.shape
        _, f_num_patch, f_embedding_dim = frozen_encoder_result.shape
        if num_patch != f_num_patch or embedding_dim != f_embedding_dim:
            raise ValueError("encoder 결과 dimension 불일치 !!")

        ## 이제 패치기준으로 embedding dim cos sim 구하기
        # torch.mul 결과 B x num_patch x embedding_dim
        dot_product = torch.sum(torch.mul(encoder_result, frozen_encoder_result), dim=-1)  # B x num_patch
        size = torch.sqrt(torch.sum(torch.mul(encoder_result, encoder_result), dim=-1))
        frozen_size = torch.sqrt(torch.sum(torch.mul(frozen_encoder_result, frozen_encoder_result), dim=-1))
        denominator = torch.mul(size, frozen_size) + self.eps  # 차원: B x num_patch
        loss_feat = 1 - torch.mean((dot_product / denominator))
        return loss_feat

    def forward(self, pred, y, len_data, disparity=False ,frozen_encoder_result=None, encoder_result=None):
        """
        :param pred: Prediction per pixel. size : BxHxW
        :param y: Ground truth. size : BxHxW
        """
        B, H, W = pred.shape
        assert len_data % 3 == 0, "len_data must be divisible by 3!!! 그래야 labeled + unlabeled + cutmix 로 나눠져 ~"
        assert len_data == B, "배치와 dataloader가 pass한 길이 불일치"

        # 두가지 종류로 나눠서 loss 구해보자
        # ssi loss를 쓰는 group_1
        # cutmix 된 group_2

        pred = pred.view(B, H * W)
        y = y.squeeze()
        y = y.view(B, H * W)

        group_1_pred = pred[:(len_data//3)*2]
        group_2_pred = pred[(len_data//3)*2:]

        # print(pred.shape)
        # print(y.shape)
        loss_u = 0

        if not disparity:
            ## 역수로 바꿔주기
            temp = torch.ones_like(y)
            y = temp / (y + self.eps)

        group_1_y = y[:(len_data//3)*2]
        group_2_y = y[((len_data//3)*2):]

        ## group_1에 대해서는 그냥 일반 ssi loss 적용해주면 ok

        group_1_y = self._normalize_depth(group_1_y)

        print("gt ( 1st element ):", group_1_y[0])
        print("pred ( 1st element ):", group_1_pred[0])
        print("y aggregation ( 1st element ):", torch.sum(group_1_y[0], dim=-1))
        print("pred aggregation ( 1st element ):", torch.sum(group_1_pred[0], dim=-1))

        loss_l = torch.sum(self._rho(group_1_pred, group_1_y), dim=-1) / group_1_pred.shape[-1]
        loss_l = loss_l.mean()
        # print(loss_l)

        ## 위에까지는 이제 전체 중 6할에 대한 ssi loss 구했음. 나머지는 cutmix 부분이므로, 새로 loss함수 설정 필요함


        # 자르기 불편하니까 다시 복귀...
        group_2_pred=group_2_pred.view(-1,H,W)
        group_2_y=group_2_y.view(-1,H,W)

        # 1. Mask 부분에 대한 loss
        pred_mask1 = group_2_pred[:,:H//2,:W//2]  ## (B x (H x W))  x (H x W) --> 이거 해보니까 element-wise하게 곱해짐 ㅇㅇ
        y_mask1 = group_2_y[:,:H//2,:W//2]

        pred_mask1 = pred_mask1.reshape(-1,H*W//4)
        y_mask1 = y_mask1.reshape(-1,H*W//4)

        y_mask1 = self._normalize_depth(y_mask1)
        loss_mask1 = torch.sum(self._rho(pred_mask1, y_mask1), dim=-1)

        # 2. (1 - Mask)에 대한 loss

        mask = torch.zeros(len_data//3,H,W)
        mask[:,:H//2,:W//2] = 1

        #print("mask: ",mask.shape)

        pred_mask2_full = group_2_pred * (1 - mask)
        y_mask2_full = group_2_y * (1 - mask)
        #print("pred_mask2: ",pred_mask2_full.shape)
        #print("bool : ", pred_mask2_full[(1 - mask).bool()].shape)

        # 유효한 픽셀들을 boolean indexing으로 추출 !!!!!
        valid_pred_mask2 = pred_mask2_full[(1 - mask).bool()].view(group_2_y.shape[0],-1)    ## 이거 배치가 4니까 그냥 view 4로 가는거야
        valid_y_mask2 = y_mask2_full[(1 - mask).bool()].view(group_2_y.shape[0],-1)

        #print(valid_pred_mask2.shape)
        #print(valid_y_mask2.shape)

        # 정규화 등 후처리를 진행하고 loss 계산
        valid_y_mask2 = self._normalize_depth(valid_y_mask2)
        loss_mask2 = torch.sum(self._rho(valid_pred_mask2, valid_y_mask2), dim=-1)

        # 3. 둘이 더하고 HW로 나눠주기
        loss_u=(loss_mask1 + loss_mask2)/group_1_pred.shape[-1]

        loss_u = loss_u.mean()

        ## 이제 labeled dataset도 빼고, 나머지 3할에 대한 부분에 feature_alignment 적용 !!

        ## 결국 인코더 결과도 B x num_patch x embedding_dim 꼴이어야 함.
        if encoder_result is None or frozen_encoder_result is None:
            raise ValueError("cos -> encoder 결과 필요")
        loss_feat = self._cosine_loss(encoder_result, frozen_encoder_result)
        if loss_feat < (1-self.threshold):
            loss_feat=0
            print("feature_loss skipped!!")

        return loss_l + loss_u + loss_feat


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
encoder_result = torch.randn(B, num_patch, embedding_dim)
frozen_encoder_result = torch.randn(B, num_patch, embedding_dim)

# Loss 모듈 인스턴스 생성
loss_module = Loss_student()
# loss 계산
total_loss = loss_module(pred, y, len_data, disparity=True, frozen_encoder_result=frozen_encoder_result, encoder_result=encoder_result)

print("Total loss:", total_loss.item())

"""