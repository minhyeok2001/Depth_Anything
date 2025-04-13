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

def train_teacher(model, hyper_params, optimizer=None, scheduler=None, scaler=None, run=None, start_epoch=0):
    from src.data.data_loader_teacher import dataloader_teacher, val_dataloader_teacher
    from src.loss.loss_teacher import Loss_teacher
    print("Running teacher training...")

    lr = hyper_params["learning_rate"]
    num_epochs = hyper_params["epochs"]
    patient = hyper_params["patient"]

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    loss_module = Loss_teacher()

    if not optimizer :
        optimizer = optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])
        
    if not scheduler :
        scheduler = ExponentialLR(optimizer, gamma=0.95)

    wandb.watch(model, log="all")

    # AMP GradScaler 생성
    if not scaler :
        scaler = GradScaler()

    best_val_loss = float('inf')
    best_epoch = 0
    trial = 0

    for epoch in range(start_epoch,num_epochs):
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
                'scheduler_state_dict': scheduler.state_dict(), 
                'scaler_state_dict': scaler.state_dict()       
            }, 'best_checkpoint_teacher.pth')
            print(f"Best checkpoint saved at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}")
            trial = 0
        else:
            trial += 1

        if trial >= patient:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best checkpoint was from epoch {best_epoch + 1} with validation loss {best_val_loss:.4f}.")
    run.finish()


def train_student(model, hyper_params, optimizer=None, scheduler=None, scaler=None, run=None, start_epoch=0):
    from src.data.data_loader_student import dataloader_student, val_dataloader_student
    from src.loss.loss_student import Loss_student
    print("Running student training...")

    lr = hyper_params["learning_rate"]
    num_epochs = hyper_params["epochs"]
    patient = hyper_params["patient"]
    threshold = hyper_params["threshold"]

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

    if not optimizer :
        optimizer = optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])

    if not scheduler :
        scheduler = ExponentialLR(optimizer, gamma=0.95)

    loss_module = Loss_student(device=device,threshold=threshold)

    wandb.watch(model, log="all")

    # AMP GradScaler 생성
    if not scaler :
        scaler = GradScaler()

    best_val_loss = float('inf')
    best_epoch = 0
    trial = 0

    for epoch in range(start_epoch, num_epochs):
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
                'scheduler_state_dict': scheduler.state_dict(), 
                'scaler_state_dict': scaler.state_dict()       
            }, 'best_checkpoint_student.pth')
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


def wandb_init(model_name=None,device=device):
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if model_name == "student":
        hyper_params = config["student_hyper_parameter"]
        run = wandb.init(project="DepthAnything_student", entity="mhroh01-ajou-university", config=hyper_params)
        model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False, student=True).to(device)
        
    elif model_name == "teacher":
        hyper_params = config["teacher_hyper_parameter"]
        run = wandb.init(project="DepthAnything_teacher", entity="mhroh01-ajou-university", config=hyper_params)
        model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False).to(device)

    return model, hyper_params, run


def load_params(model,checkpoint,hyper_params):

    optimizer = optim.AdamW(model.parameters(), lr=hyper_params["learning_rate"])
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    scaler = GradScaler()

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    start_epoch = checkpoint['epoch']

    return model, optimizer, scheduler, scaler, start_epoch



def wandb_resume(model_name=None,device=device,run_id=None):
    config_path = "configs/config.yaml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if model_name == "student":
        hyper_params = config["student_hyper_parameter"]
        run = wandb.init(project="DepthAnything_student", entity="mhroh01-ajou-university", config=hyper_params,resume="allow",id=run_id)
        model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False, student=True).to(device)
        checkpoint = torch.load("best_checkpoint_student.pth")
        model, optimizer, scheduler, scaler, start_epoch = load_params(model,checkpoint,hyper_params)
        print(f"Resumed student from epoch {start_epoch}")
        
    elif model_name == "teacher":
        hyper_params = config["teacher_hyper_parameter"]
        run = wandb.init(project="DepthAnything_teacher", entity="mhroh01-ajou-university", config=hyper_params,resume="allow",id=run_id)
        model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024], use_bn=False, localhub=False).to(device)
        checkpoint = torch.load("best_checkpoint_teacher.pth")
        model ,optimizer, scheduler, scaler, start_epoch = load_params(model,checkpoint,hyper_params)
        print(f"Resumed teacher from epoch {start_epoch}")

    return model, hyper_params, optimizer, scheduler, scaler, run, start_epoch

def main():
    parser = argparse.ArgumentParser(description="Train teacher or student model")
    parser.add_argument('--teacher', action='store_true', help="Run teacher training")
    parser.add_argument('--student', action='store_true', help="Run student training")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run_id',type=str, help="wand run_id (required only if --resume is used)" )
    # parser 객체 생성시, 자동으로 -- 같은 dash는 없어지고 내부 인스턴스 호출 가능
    args = parser.parse_args()

    if args.teacher and args.student:
        raise ValueError("Cannot specify both --teacher and --student.")

    elif not args.teacher and not args.student:
        raise ValueError("Invalid mode")

    default_function()

    if args.resume :
        if args.run_id is None:
            raise ValueError("--run_id is required when using --resume")
        else :
            if args.teacher:
                model, hyper_params, optimizer, run, scheduler, scaler, start_epoch = wandb_resume(model_name="teacher",device=device,run_id=args.run_id)
                train_teacher(model,hyper_params,optimizer,scheduler, scaler,run,start_epoch)
            elif args.student:
                model, hyper_params, optimizer, run, scheduler, scaler, start_epoch = wandb_resume(model_name="student",device=device,run_id=args.run_id)
                train_student(model,hyper_params,optimizer,scheduler, scaler,run,start_epoch)
        
    else :  
        if args.teacher:
            model, hyper_params, run= wandb_init(model_name="teacher",device=device)
            train_teacher(model=model,hyper_params=hyper_params,run=run)
        elif args.student:
            model, hyper_params, run= wandb_init(model_name="student",device=device)
            train_student(model=model,hyper_params=hyper_params,run=run)
    
if __name__ == "__main__":
    main()
