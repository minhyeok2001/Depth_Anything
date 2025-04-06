import os
import glob
from PIL import Image
import imageio.v2 as imageio
from torchvision import transforms
from dotenv import load_dotenv,find_dotenv
from torch.utils.data import DataLoader
from src.data.load_data import get_data_list, customDataset
import yaml
import torch

config_path = "configs/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

hyper_params = config["teacher_hyper_parameter"]
batch_size = hyper_params["batch_size"]
train_data_size = hyper_params["train_data_size"]
val_data_size = hyper_params["val_data_size"]

# env로부터 경로 가져오고, 인풋이미지와 gt 이미지 경로만 로드
load_dotenv(find_dotenv())
dataset_path = os.getenv("HRWSI_DATASET_PATH")

x_path_list , gt_path_list= get_data_list(dataset_path,data_name="hrwsi",end_idx=train_data_size)
dataset = customDataset(x_path_list,gt_path_list,transform=False)

val_x_path_list , val_gt_path_list= get_data_list(dataset_path,data_name="hrwsi",end_idx=val_data_size)
val_dataset = customDataset(val_x_path_list,val_gt_path_list,transform=False)
#print("Current working directory:", os.getcwd())

dataloader_teacher = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader_teacher = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

"""
for x,y in dataloader:
    print(x)
    print(y)
    break
# 잘 나오는 것 확인했음
"""
