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

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} from {checkpoint_path}")
    return model

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
            if not visualized:
                input_img = inputs[0].cpu()
                pred_depth = outputs_corr[0].cpu()
                input_img_np = input_img.permute(1, 2, 0).numpy()
                input_img_np = np.clip(input_img_np, 0, 1)
                pred_depth_np = pred_depth.numpy()
                min_val = np.min(pred_depth_np)
                max_val = np.max(pred_depth_np)
                normalized_depth = (pred_depth_np - min_val) / (max_val - min_val + 1e-8)
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(normalized_depth, cmap='gray')
                ax.axis('off')
                ax.set_title('Predicted Depth (1-channel)')
                plt.tight_layout()
                plt.show()
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
            if not visualized:
                input_img = inputs[0].cpu()
                pred_depth = outputs_corr[0].cpu()
                input_img_np = input_img.permute(1, 2, 0).numpy()
                input_img_np = np.clip(input_img_np, 0, 1)
                pred_depth_np = pred_depth.numpy()
                min_val = np.min(pred_depth_np)
                max_val = np.max(pred_depth_np)
                normalized_depth = (pred_depth_np - min_val) / (max_val - min_val + 1e-8)
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                ax.imshow(normalized_depth, cmap='gray')
                ax.axis('off')
                ax.set_title('Predicted Depth (1-channel)')
                plt.tight_layout()
                plt.show()
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