import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm

from dotenv import load_dotenv, find_dotenv

from src.utils.check_device import device
from src.utils.metrics import scale_shift_correction, compute_abs_rel, compute_delta1
from src.models.model import DepthModel
from src.loss.loss_teacher import Loss_teacher
from src.loss.loss_student import Loss_student

load_dotenv(find_dotenv())

# OpenMP 런타임 중복 문제 우회를 위한 임시 해결책
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} from {checkpoint_path}")
    return model

def visualize_sample(original_tensor, gt_tensor, pred_tensor, title_prefix=""):
    """
    original_tensor: Tensor of shape (C, H, W)
    gt_tensor, pred_tensor: Tensor of shape (1, H, W) 혹은 (H, W)
    """
    # 원본 이미지: RGB일 경우 (3, H, W) -> (H, W, 3)로 변환, Grayscale일 경우 (1, H, W) -> (H, W)
    original_np = original_tensor.cpu().numpy()
    if original_np.shape[0] == 1:
        original_np = original_np.squeeze(0)
    else:
        original_np = np.transpose(original_np, (1, 2, 0))
    original_np = np.clip(original_np, 0, 1)

    # Ground Truth Depth 처리: tensor가 (1, H, W)인 경우 squeeze해서 (H,W)
    gt_np = gt_tensor.cpu().numpy()
    if gt_np.ndim == 3 and gt_np.shape[0] == 1:
        gt_np = np.squeeze(gt_np, axis=0)
    min_gt, max_gt = np.min(gt_np), np.max(gt_np)
    normalized_gt = (gt_np - min_gt) / (max_gt - min_gt + 1e-8)

    # 예측 Depth 처리: tensor가 (1, H, W)인 경우 squeeze해서 (H,W)
    pred_np = pred_tensor.cpu().numpy()
    if pred_np.ndim == 3 and pred_np.shape[0] == 1:
        pred_np = np.squeeze(pred_np, axis=0)
    min_pred, max_pred = np.min(pred_np), np.max(pred_np)
    normalized_pred = (pred_np - min_pred) / (max_pred - min_pred + 1e-8)

    # 3개의 이미지를 한 화면에 출력
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_np, cmap=None)
    axes[0].axis('off')
    axes[0].set_title(f'{title_prefix}Original Image')
    
    axes[1].imshow(normalized_gt, cmap='gray')
    axes[1].axis('off')
    axes[1].set_title(f'{title_prefix}Ground Truth Depth')
    
    axes[2].imshow(normalized_pred, cmap='gray')
    axes[2].axis('off')
    axes[2].set_title(f'{title_prefix}Predicted Depth')
    
    plt.tight_layout()
    plt.show()

def predict_teacher():
    from src.data.data_loader_teacher import val_dataloader_teacher
    print("Running teacher prediction...")
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    hyper_params = config["teacher_hyper_parameter"]
    model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False).to(device)
    checkpoint_path = "best_checkpoint_teacher.pth"
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    loss_module = Loss_teacher()
    running_val_loss = 0.0
    abs_rel_total = 0.0
    delta1_total = 0.0
    flag=0
    visualized = False
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader_teacher):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs_corr = scale_shift_correction(outputs, targets)
            loss = loss_module(outputs, targets, disparity=True)
            running_val_loss += loss.item()
            abs_rel_total += compute_abs_rel(outputs_corr, targets)
            delta1_total += compute_delta1(outputs_corr, targets)
            
            # 첫번째 배치에서 원본, GT, 예측을 시각화
            if not visualized:
                visualize_sample(inputs[0], targets[0], outputs_corr[0], title_prefix="Teacher: ")
                flag+=1
                if flag> 8:
                    visualized = True
                
    num_batches = len(val_dataloader_teacher)
    print(f"Avg Validation Loss: {running_val_loss / num_batches:.4f}")
    print(f"Avg AbsRel: {abs_rel_total / num_batches:.4f}")
    print(f"Avg Delta1: {delta1_total / num_batches:.4f}")

def predict_student():
    from src.data.data_loader_student import val_dataloader_student
    print("Running student prediction...")
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    hyper_params = config["student_hyper_parameter"]
    model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False, student=True).to(device)
    checkpoint_path = "best_checkpoint_student.pth"
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    frozen_encoder = "vitb"
    frozen_model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{frozen_encoder}14', pretrained=True)
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False
    frozen_model = frozen_model.to(device)
    loss_module = Loss_student(device=device, threshold=hyper_params["threshold"])
    running_val_loss = 0.0
    abs_rel_total = 0.0
    delta1_total = 0.0
    visualized = False
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_dataloader_student):
            B = inputs.shape[0]
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, burning_feature = model(inputs)
            frozen_features = frozen_model.get_intermediate_layers(inputs[B // 3:(B // 3) * 2], n=1, return_class_token=False)
            loss = loss_module(outputs, targets, len_data=inputs.shape[0], disparity=True,
                               frozen_encoder_result=frozen_features[0], encoder_result=burning_feature)
            outputs_corr = scale_shift_correction(outputs, targets)
            running_val_loss += loss.item()
            abs_rel_total += compute_abs_rel(outputs_corr, targets)
            delta1_total += compute_delta1(outputs_corr, targets)
            
            # 첫번째 배치에서 원본, GT, 예측을 시각화
            if not visualized:
                visualize_sample(inputs[0], targets[0], outputs_corr[0], title_prefix="Student: ")
                flag +=1
                if flag > 8:
                    visualized = True
                
    num_batches = len(val_dataloader_student)
    print(f"Avg Validation Loss: {running_val_loss / num_batches:.4f}")
    print(f"Avg AbsRel: {abs_rel_total / num_batches:.4f}")
    print(f"Avg Delta1: {delta1_total / num_batches:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Predict teacher or student model")
    parser.add_argument('--teacher', action='store_true', help="Run teacher prediction")
    parser.add_argument('--student', action='store_true', help="Run student prediction")
    args = parser.parse_args()
    if args.teacher and args.student:
        raise ValueError("Cannot specify both --teacher and --student.")
    elif not args.teacher and not args.student:
        raise ValueError("Invalid mode. Please specify either --teacher or --student.")
    if args.teacher:
        predict_teacher()
    elif args.student:
        predict_student()

if __name__ == "__main__":
    main()