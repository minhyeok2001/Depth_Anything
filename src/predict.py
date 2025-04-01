import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from src.data.data_loader_teacher import val_dataloader
from src.models.model import DepthModel
from src.loss.loss_teacher import teacher_loss_function
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 때와 동일한 파라미터로 모델 생성 및 checkpoint 불러오기
model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=True, localhub=False).to(device)
checkpoint_path = 'final_checkpoint.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 추론 모드로 전환

running_val_loss = 0.0

# val_dataloader에서 한 배치만 사용하여 결과 확인 (여러 배치에 대해 반복 가능)
with torch.no_grad():
    for idx,(inputs, targets) in enumerate(val_dataloader):
        if idx == 0:
            continue
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)  # outputs: (B, H, W)
        loss = teacher_loss_function(outputs, targets, disparity=False)
        running_val_loss += loss.item()
        
        # 시각화를 위해 배치에서 첫 번째 샘플 사용
        # 원본 이미지: inputs[0] -> (C, H, W)
        # 예측 깊이: outputs[0] -> (H, W)
        input_img = inputs[0].cpu()
        pred_depth = outputs[0].cpu()
        
        # 원본 이미지 처리
        # (만약 입력 이미지가 정규화(normalize) 되어 있다면, unnormalize 과정이 필요합니다.)
        # 여기서는 입력이 [0, 1] 범위라고 가정하고 진행합니다.
        input_img_np = input_img.permute(1, 2, 0).numpy()  # (H, W, C)
        input_img_np = np.clip(input_img_np, 0, 1)  # 값 범위 보정
        
        # 예측 깊이에 대해 min-max normalization
        pred_depth_np = pred_depth.numpy()
        min_val = np.min(pred_depth_np)
        max_val = np.max(pred_depth_np)
        normalized_depth = (pred_depth_np - min_val) / (max_val - min_val + 1e-8)
        
        # 컬러맵 적용 (inferno 등)
        cmap = plt.get_cmap('inferno')
        depth_colormap = cmap(normalized_depth)  # RGBA 이미지
        depth_colormap = (depth_colormap[..., :3] * 255).astype(np.uint8)  # RGB로 변환
        
        # 시각화: 원본 이미지와 예측 깊이 이미지를 나란히 출력
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(input_img_np)
        axes[0].set_title('Original Input')
        axes[0].axis('off')
        
        axes[1].imshow(depth_colormap)
        axes[1].set_title('Predicted Depth (min-max normalized)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 한 배치만 확인 (여러 배치를 확인하려면 break 제거)
        break