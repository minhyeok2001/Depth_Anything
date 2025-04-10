import os
import glob
from PIL import Image
import imageio.v2 as imageio
from torchvision import transforms
from dotenv import load_dotenv,find_dotenv
from torch.utils.data import DataLoader
from src.data.load_data import get_data_list, customDataset, combinedDataset, mixedBatchSampler,cutMix_collate_fn
import yaml
import numpy as np

#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

config_path = "configs/config.yaml"
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

hyper_params = config["student_hyper_parameter"]
batch_size = hyper_params["batch_size"]
labeled_train_data_size = hyper_params["train_data_size"]
pseudo_train_data_size = labeled_train_data_size*2  ## 이거는 당연히 config에서 설정할 때, labeled의 절반으로 해야지
labeled_val_data_size = hyper_params["val_data_size"]
pseudo_val_data_size = labeled_val_data_size*2
# env로부터 경로 가져오고, 인풋이미지와 gt 이미지 경로만 로드

assert labeled_train_data_size % 3 == 0, " 데이터 무조건 3으로 나눠져야함 !! "
assert pseudo_val_data_size % 3 == 0, " 데이터 무조건 3으로 나눠져야함 !! "

load_dotenv(find_dotenv())
labeled_dataset_path = os.getenv("HRWSI_DATASET_PATH")
pseudo_labeled_dataset_path = os.getenv("GOOGLELANDMARK_DATASET_PATH")

#labeled_dataset_path = "/Users/minhyeokroh/Documents/2025-1/DA_dataset/HR-WSI"
#pseudo_labeled_dataset_path = "/Users/minhyeokroh/Documents/2025-1/DA_dataset/blendedMVS"

x_path_list , gt_path_list= get_data_list(labeled_dataset_path,data_name="hrwsi",end_idx=labeled_train_data_size)
labeled_dataset = customDataset(x_path_list,gt_path_list,transform=False)

pseudo_x_path_list , pseudo_gt_path_list = get_data_list(pseudo_labeled_dataset_path,data_name="google_landmark",end_idx=pseudo_train_data_size)
pseudo_dataset = customDataset(pseudo_x_path_list,pseudo_gt_path_list,transform=True)

combined_dataset = combinedDataset(labeled_dataset,pseudo_dataset)
mix_sampler= mixedBatchSampler(labeled_dataset,pseudo_dataset,batch_size,labeled_ratio=0.334)
dataloader_student = DataLoader(combined_dataset,batch_sampler=mix_sampler,collate_fn=cutMix_collate_fn)


x_val_path_list, gt_val_path_list = get_data_list(labeled_dataset_path, val=True, data_name="hrwsi", end_idx=labeled_val_data_size)
labeled_val_dataset = customDataset(x_val_path_list, gt_val_path_list, transform=False)

pseudo_x_val_path_list, pseudo_gt_val_path_list = get_data_list(pseudo_labeled_dataset_path, val=True, data_name="google_landmark", end_idx=pseudo_val_data_size)
pseudo_val_dataset = customDataset(pseudo_x_val_path_list, pseudo_gt_val_path_list, transform=True)

# Validation 데이터를 위한 합친 데이터셋 생성 및 배치 샘플러 구성
combined_val_dataset = combinedDataset(labeled_val_dataset, pseudo_val_dataset)
mix_val_sampler = mixedBatchSampler(labeled_val_dataset, pseudo_val_dataset, batch_size, labeled_ratio=0.334)

# DataLoader 구성 (collate_fn은 train과 동일하게 cutMix_collate_fn 사용)
val_dataloader_student = DataLoader(combined_val_dataset, batch_sampler=mix_val_sampler, collate_fn=cutMix_collate_fn)


"""
for x,y in val_dataloader_student:
    print(x)
    break

import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import ToPILImage

to_pil = ToPILImage()
for x, gt in dataloader_student:
    batch_size = x.shape[0]
    print("B S : ",x.shape[0])
    fig, axes = plt.subplots(nrows=2, ncols=batch_size, figsize=(3*batch_size, 6))

    if batch_size == 1:
        axes = [[axes[0]], [axes[1]]]

    for i in range(batch_size):
        # tensor 
        axes[0][i].imshow(to_pil(x[i]))
        axes[0][i].set_title("Input Image")
        axes[0][i].axis('off')

        axes[1][i].imshow(to_pil(gt[i]))
        axes[1][i].set_title("GT Image")
        axes[1][i].axis('off')

    plt.tight_layout()
    plt.show()
    break
"""
    
