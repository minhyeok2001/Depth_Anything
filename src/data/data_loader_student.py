import os
import glob
from PIL import Image
import imageio.v2 as imageio
from torchvision import transforms
from dotenv import load_dotenv,find_dotenv
from torch.utils.data import DataLoader
from src.data.load_data import get_data_list, customDataset
import yaml

## student_hyper_parameter

config_path = "configs/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

hyper_params = config["student_hyper_parameter"]
batch_size = hyper_params["batch_size"]
train_data_size = hyper_params["train_data_size"]
pseudo_train_data_size = hyper_params["pseudo_data_size"]  ## 이거는 당연히 config에서 설정할 때, labeled의 절반으로 해야지
val_data_size = hyper_params["val_data_size"]
pseudo_val_data_size = hyper_params["pseudo_val_data_size"] ## 이거는 당연히 config에서 설정할 때, labeled의 절반으로 해야지
# env로부터 경로 가져오고, 인풋이미지와 gt 이미지 경로만 로드

load_dotenv(find_dotenv())
labeled_dataset_path = os.getenv("BLENDEDMVS_DATASET_PATH")
pseudo_labeled_dataset_path = os.getenv("GOOGLE_DATASET_PATH")

x_path_list , gt_path_list= get_data_list(labeled_dataset_path,end_idx=train_data_size)
labeled_dataset = customDataset(x_path_list,gt_path_list,transform=False)

pseudo_x_path_list , pseudo_gt_path_list= get_data_list(pseudo_labeled_dataset_path,end_idx=pseudo_train_data_size)
pseudo_dataset = customDataset(pseudo_x_path_list,pseudo_gt_path_list,transform=True)


val_x_path_list , val_gt_path_list = get_data_list(labeled_dataset_path,start_idx=train_data_size,end_idx=train_data_size+val_data_size)
val_dataset = customDataset(val_x_path_list,val_gt_path_list,transform=False)

pseudo_val_x_path_list , pseudo_val_gt_path_list = get_data_list(pseudo_labeled_dataset_path,start_idx=pseudo_train_data_size,end_idx=pseudo_train_data_size+pseudo_val_data_size)
pseudo_val_dataset = customDataset(pseudo_val_x_path_list ,pseudo_val_gt_path_list ,transform=False)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

