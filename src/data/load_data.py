## 여기서 리스트 가져오는 함수를 하나 만들기. 

import os
import glob
import torch
from torchvision import transforms
from PIL import Image

def get_data_list(dataset_path,teacher=True,start_idx=0,end_idx=1600):
    if teacher:
        input_image_paths = sorted(glob.glob(os.path.join(dataset_path, "*", "blended_images", "*_masked.jpg")))[start_idx:end_idx]
        gt_image_paths = sorted(glob.glob(os.path.join(dataset_path, "*", "rendered_depth_maps", "*.pfm")))[start_idx:end_idx]
    
    else :
        input_image_paths = sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))
        gt_image_paths = sorted(glob.glob(os.path.join(dataset_path, "pseudo_depth_maps", "*.pfm")))    ## teacher model로부터 만들어지는 pseudo map

    #print(len(input_image_paths))
    return input_image_paths, gt_image_paths


# 커스텀 데이터셋 생성
class customDataset(torch.utils.data.Dataset):
    def __init__(self,x_path,gt_path,transform=True):
        super().__init__()
        self.x_path = x_path
        self.gt_path = gt_path
        self.transform = transform
        self.basic_transformation = transforms.Compose([
            transforms.CenterCrop(448),
            transforms.ToTensor()
        ])

        self.transformation_for_pseudo_labeled = transforms.Compose([
            transforms.CenterCrop(448),
            transforms.jitter(),
            transforms.gaussian_blur(),
            transforms.ToTensor()
        ])

    def __len__(self):
        if len(self.x_path) == len(self.gt_path):
            return len(self.x_path)
        else:
            assert "x_path와 gt_path의 길이 불일치 !!"

    def __getitem__(self,idx):
        # 지금까지 해왓던 일반적인 방식과는 다르게, 여기서 데이터를 "직접 로드"하고 리턴하는 식으로 진행
        x = Image.open(self.x_path[idx]).convert('RGB')
        gt = Image.open(self.gt_path[idx]).convert('F')

        if self.transform :
            x = self.transformation_for_pseudo_labeled(x)
            gt = self.basic_transformation(gt)

        else :
            x = self.basic_transformation(x)
            gt = self.basic_transformation(gt)

        return x, gt