import os
import glob
from PIL import Image
import imageio.v2 as imageio
from torchvision import transforms
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from load_data import get_data_list, customDataset
import yaml

# env로부터 경로 가져오고, 인풋이미지와 gt 이미지 경로만 로드
load_dotenv(dotenv_path="../../.env")
dataset_path = os.getenv("BLENDEDMVS_DATASET_PATH")

x_path_list , gt_path_list= get_data_list(dataset_path)
dataset = customDataset(x_path_list,gt_path_list,transform=False)

config_path = "path/to/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

hyper_params = config["hyper_parameter"]
batch_size = hyper_params["batch_size"]

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

