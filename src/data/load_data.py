## 여기서 리스트 가져오는 함수를 하나 만들기. 

import os
import glob
import torch
from torchvision import transforms
from PIL import Image
import torch
import random
import numpy as np

def get_data_list(dataset_path,data_name,val=False,start_idx=0,end_idx=16000):
    """
    :param dataset_path:  .env로부터 각자 데이터셋 로드해오기
    :data_name : hrwsi인지, google_landmark 인지, NYUv2인지 알아야 path를 좀 specific 하게 정할듯 ...
    :return:
    """
    if data_name == "hrwsi":
        if val :
            input_image_paths = sorted(glob.glob(os.path.join(dataset_path,"val", "imgs", "*.jpg")))[start_idx:end_idx]
            gt_image_paths = sorted(glob.glob(os.path.join(dataset_path, "val", "gts", "*.png")))[start_idx:end_idx]
        else :
            input_image_paths = sorted(glob.glob(os.path.join(dataset_path, "train", "imgs", "*.jpg")))[start_idx:end_idx]
            gt_image_paths = sorted(glob.glob(os.path.join(dataset_path, "train", "gts", "*.png")))[start_idx:end_idx]

    elif data_name == "google_landmark":
        if val :
            input_image_paths = sorted(glob.glob(os.path.join(dataset_path,"val", "image" ,"*.jpg")))[start_idx:end_idx]
            gt_image_paths = sorted(glob.glob(os.path.join(dataset_path, "val", "pseudo_depth", "*.npy")))[start_idx:end_idx]
        else :
            input_image_paths = sorted(glob.glob(os.path.join(dataset_path,"image" ,"*.jpg")))[start_idx:end_idx]
            gt_image_paths = sorted(glob.glob(os.path.join(dataset_path, "pseudo_depth", "*.npy")))[start_idx:end_idx]  ## teacher model로부터 만들어지는 pseudo -> npy tensor

    elif data_name == "bleneded_mvs":
        input_image_paths = sorted(glob.glob(os.path.join(dataset_path,"blended_images" ,"*_masked.jpg")))[start_idx:end_idx]
        gt_image_paths = sorted(glob.glob(os.path.join(dataset_path, "rendered_depth_maps", "*.pfm")))[start_idx:end_idx]

    #"to be implemented... 새로 데이터셋 추가하는대로 계속 아래에 path 추가"

    else:
        raise ValueError("Invalid data name")

    return input_image_paths, gt_image_paths


# 커스텀 데이터셋 생성 -> handing npy added !!
class customDataset(torch.utils.data.Dataset):
    def __init__(self, x_path, gt_path, transform=False):
        super().__init__()
        self.x_path = x_path
        self.gt_path = gt_path
        self.transform = transform
        self.basic_transformation = transforms.Compose([
            transforms.CenterCrop(448),
            transforms.ToTensor()
        ])
        self.transformation_for_pseudo_label = transforms.Compose([
            transforms.CenterCrop(448),
            transforms.ColorJitter(brightness=.5, hue=.3),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(5.)),
            transforms.ToTensor()
        ])

    def __len__(self):
        if len(self.x_path) == len(self.gt_path):
            return len(self.x_path)
        else:
            raise ValueError("x_path와 gt_path의 길이가 다릅니다.")

    def __getitem__(self, idx):
        x = Image.open(self.x_path[idx]).convert('RGB')
    
        gt_path = self.gt_path[idx]
        if gt_path.lower().endswith('.npy'):
            gt = np.load(gt_path)
            if gt.ndim == 2:
                gt = np.expand_dims(gt, axis=0)
            gt = torch.from_numpy(gt).float()
        else:
            gt = Image.open(gt_path).convert('F')
            gt = self.basic_transformation(gt)

        if self.transform:
            x = self.transformation_for_pseudo_label(x)
        else :
            x = self.basic_transformation(x)

        return x, gt

class combinedDataset(torch.utils.data.Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset

    def __getitem__(self, index_tuple):
        # index_tuple >>> (is_labeled, idx)
        is_labeled, idx = index_tuple
        if is_labeled:
            return self.labeled_dataset[idx]
        else:
            return self.unlabeled_dataset[idx]

    def __len__(self):
        # 실제 사용은 sampler가 배치 인덱스를 제공하므로 __len__은 크게 중요하지 않음
        return len(self.labeled_dataset) + len(self.unlabeled_dataset)


class mixedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labeled_dataset, unlabeled_dataset, batch_size, labeled_ratio):

        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.batch_size = batch_size  # 12
        self.labeled_per_batch = int(batch_size * labeled_ratio)  # 8
        self.unlabeled_per_batch = batch_size - self.labeled_per_batch  # 4

        #print("labeled num: ",self.labeled_per_batch)
        #print("unlabeled num: ",self.unlabeled_per_batch)

    def __iter__(self):
        labeled_indices = list(range(len(self.labeled_dataset)))     ## 이부분은 실제 데이터를 받아서 len을 하는게 아니라, utils.data.dataset을 상속받은 클래스 내부에 __len__ 함수가 있으므로 그걸 가져옴
        unlabeled_indices = list(range(len(self.unlabeled_dataset)))
        random.shuffle(labeled_indices)
        random.shuffle(unlabeled_indices)

        num_batches = min(len(labeled_indices) // self.labeled_per_batch,len(unlabeled_indices) // self.unlabeled_per_batch)

        for i in range(num_batches):
            batch = []
            start_l = i * self.labeled_per_batch
            start_u = i * self.unlabeled_per_batch
            for idx in labeled_indices[start_l: start_l + self.labeled_per_batch]:
                batch.append((True, idx))
            for idx in unlabeled_indices[start_u: start_u + self.unlabeled_per_batch]:
                batch.append((False, idx))
            yield batch
            ## 이러면 여기서 인덱스 배치를 리턴하게 됨

    def __len__(self):
        labeled_batches = len(self.labeled_dataset) // self.labeled_per_batch
        unlabeled_batches = len(self.unlabeled_dataset) // self.unlabeled_per_batch
        return min(labeled_batches, unlabeled_batches)

def cutMix_collate_fn(data):
    """
    참고로 6의 배수로 배치사이즈를 잡는 것이 좋아보임 ㅎㅎ... 그래야 딱 나눠떨어지고, 컷믹스 쌍도 딱 맞음
    :param batch_data: 내부적으로 동작하는거라 실제로 뽑아보지는 못했으나, (input_image,GT_image) 쌍이 배치사이즈만큼의 크기로 리스트형태로 들어왔을 것임
    :return:
    """
    ## 어차피 큰수의 법칙에 따라서, 1/2 확률로 샘플링 하는거면 그냥 데이터 절반만 cutmix 적용시켜도 무방하다고 생각함.
    ratio = 0.334
    len_data = len(data)
    #print("batchsize :",len_data)
    num_cutMixImg= int(len_data * ratio) # 컷믹스를 진행할 이미지 개수
    print(num_cutMixImg)

    ## 맘 같아서는 어차피 crop 448x448이니까 그걸로 하고싶지만, 이건 바뀔 수도 있으므로 일단 standby
    c, x_shape, y_shape = data[0][0].shape #448 기대중

    ## 마스크는 shape 기준으로 1/2, 1/2 부분. 즉 1/4만 씌워주도록 설해보자 -> 논문에서는 Mask가 실제로 어떻게 생겼는지에 대한 언급은 없음
    mask = torch.zeros(data[0][0][0].shape)
    
    # print(mask)
    mask[0:x_shape//2,0:y_shape//2]=1
    #print(mask)
    #print(1-mask)
    #print(type(data))
    ## 아아아 ..이게 data는 리스트이지만, 이 안에 데이터는 텐서지 ....
    temp_x = [item[0] for item in data[:num_cutMixImg*2]]
    temp_y = [item[1] for item in data[:num_cutMixImg*2]]

    for i in range(num_cutMixImg * 2, len_data, 2):
        #print(data[i][0])
        #print(data[i][0].shape)
        #print(mask)
        temp_x.append(data[i][0] * mask + data[i+1][0] * (1 - mask))
        temp_x.append(data[i][0] * (1 - mask) + data[i+1][0] * mask)
        temp_y.append(data[i][1] * mask + data[i+1][1] * (1 - mask))
        temp_y.append(data[i][1] * (1 - mask) + data[i+1][1] * mask)

        ## 이거 일단 gt도 섞어서 위처럼 주고, loss 구할때 발라서 먹자

    ## temp는 이제 stack 해야함 ..

    x = torch.stack(temp_x) # B x X
    y = torch.stack(temp_y) # B x Y ( 1 x H X W )

    return x,y
