import os
import glob
import argparse
import yaml

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from dotenv import load_dotenv, find_dotenv

from src.utils.check_device import device
from src.utils.metrics import scale_shift_correction, compute_abs_rel, compute_delta1
from src.models.model import DepthModel

def train_teacher():
    from src.data.data_loader_teacher import dataloader_teacher, val_dataloader_teacher
    from src.loss.loss_teacher import Loss_teacher
    print("Running teacher training...")

    # hyper params
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    hyper_params = config["teacher_hyper_parameter"]
    lr = hyper_params["learning_rate"]
    num_epochs = hyper_params["epochs"]
    patient = hyper_params["patient"]

    run = wandb.init(project="DepthAnything_teacher", entity="mhroh01-ajou-university", config=hyper_params)

    model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    loss_module = Loss_teacher()

    wandb.watch(model, log="all")

    # AMP GradScaler 생성
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_epoch = 0
    trial = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader_teacher)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                outputs = model(inputs)
                loss = loss_module(outputs, targets, disparity=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader_teacher)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")
        scheduler.step()

        model.eval()
        running_val_loss = 0.0
        abs_rel = 0.0
        delta1 = 0.0

        with torch.no_grad():
            for inputs, targets in val_dataloader_teacher:
                inputs, targets = inputs.to(device), targets.to(device)
           
                outputs = model(inputs)
                loss = loss_module(outputs, targets, disparity=True)
                outputs = scale_shift_correction(outputs, targets)
                running_val_loss += loss.item()
                abs_rel += compute_abs_rel(outputs, targets)
                delta1 += compute_delta1(outputs, targets)

        avg_val_loss = running_val_loss / len(val_dataloader_teacher)
        avg_abs_rel = abs_rel / len(val_dataloader_teacher)
        avg_delta1 = delta1 / len(val_dataloader_teacher)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] Abs Rel: {avg_abs_rel:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] Delta1: {avg_delta1:.4f}")
        wandb.log({
            "train_loss": epoch_loss,
            "val_loss": avg_val_loss,
            "abs_rel": avg_abs_rel,
            "delta1": avg_delta1,
            "epoch": epoch + 1
        })

        # 베스트 체크포인트 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'best_checkpoint_teahcer.pth')
            print(f"Best checkpoint saved at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}")
            trial = 0
        else:
            trial += 1

        if trial >= patient:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best checkpoint was from epoch {best_epoch + 1} with validation loss {best_val_loss:.4f}.")
    run.finish()


def train_student():
    from src.data.data_loader_student import dataloader_student, val_dataloader_student
    from src.loss.loss_student import Loss_student
    print("Running student training...")

    # hyper params
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    hyper_params = config["student_hyper_parameter"]
    lr = hyper_params["learning_rate"]
    num_epochs = hyper_params["epochs"]
    patient = hyper_params["patient"]
    threshold = hyper_params["threshold"]

    run = wandb.init(project="DepthAnything_student", entity="mhroh01-ajou-university", config=hyper_params)

    model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False, student=True).to(device)

    frozen_encoder = "vitb"
    frozen_model = torch.hub.load(
        'facebookresearch/dinov2',
        f'dinov2_{frozen_encoder}14',
        pretrained=True
    )
    frozen_model.eval()
    for param in frozen_model.parameters():
        param.requires_grad = False  # 파라미터 업데이트 안되도록 고정
    frozen_model = frozen_model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    loss_module = Loss_student(device=device,threshold=threshold)

    wandb.watch(model, log="all")

    # AMP GradScaler 생성
    scaler = GradScaler()

    best_val_loss = float('inf')
    best_epoch = 0
    trial = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in tqdm(enumerate(dataloader_student)):
            B = inputs.shape[0]
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                outputs, burning_feature = model(inputs)
                frozen_feature = frozen_model.get_intermediate_layers(inputs[B//3:(B//3)*2],n=1,return_class_token=False)
                #print("frozen feature : ", frozen_feature.shape) 아!! 이게 CLS token 이었구나 !!! 이미지 하나 넣으면 cls 토큰 하나를 뽑아주는거야 !!
                loss = loss_module(outputs, targets,len_data=(inputs.shape[0]),disparity=True,frozen_encoder_result=frozen_feature[0], encoder_result=burning_feature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(dataloader_student)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")
        scheduler.step()

        model.eval()
        running_val_loss = 0.0
        abs_rel = 0.0
        delta1 = 0.0

        with torch.no_grad():
            for inputs, targets in val_dataloader_student:
                B = inputs.shape[0]
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, burning_feature = model(inputs)
                frozen_feature = frozen_model.get_intermediate_layers(inputs[B // 3:(B // 3) * 2], n=1,return_class_token=False)
                loss = loss_module(outputs, targets, len_data=(inputs.shape[0]), disparity=True,frozen_encoder_result=frozen_feature[0], encoder_result=burning_feature)
                outputs = scale_shift_correction(outputs, targets)
                running_val_loss += loss.item()
                abs_rel += compute_abs_rel(outputs, targets)
                delta1 += compute_delta1(outputs, targets)

        avg_val_loss = running_val_loss / len(val_dataloader_student)
        avg_abs_rel = abs_rel / len(val_dataloader_student)
        avg_delta1 = delta1 / len(val_dataloader_student)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] Abs Rel: {avg_abs_rel:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] Delta1: {avg_delta1:.4f}")

        wandb.log({
            "train_loss": epoch_loss,
            "val_loss": avg_val_loss,
            "abs_rel": avg_abs_rel,
            "delta1": avg_delta1,
            "epoch": epoch + 1
        })

        # 베스트 체크포인트 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f'best_checkpoint_student.pth')
            print(f"Best checkpoint saved at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}")
            trial = 0
        else:
            trial += 1

        if trial >= patient:
            print("Early stopping triggered.")
            break

    print(
        f"Training finished. Best checkpoint was from epoch {best_epoch + 1} with validation loss {best_val_loss:.4f}.")
    run.finish()

def default_function(device=device):
    # GPU check
    device = device

    load_dotenv(find_dotenv())
    api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=api_key)



def main():
    parser = argparse.ArgumentParser(description="Train teacher or student model")
    parser.add_argument('--teacher', action='store_true', help="Run teacher training")
    parser.add_argument('--student', action='store_true', help="Run student training")
    args = parser.parse_args()


    if args.teacher and args.student:
        raise ValueError("Cannot specify both --teacher and --student.")

    elif not args.teacher and not args.student:
        raise ValueError("Invalid mode")

    default_function()

    if args.teacher:
        train_teacher()
    elif args.student:
        train_student()

if __name__ == "__main__":
    main()
