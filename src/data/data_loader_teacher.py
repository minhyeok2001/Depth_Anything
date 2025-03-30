import os
import glob
from PIL import Image
import imageio.v2 as imageio
from torchvision import transforms
from dotenv import load_dotenv
from torch.utils.data import TensorDataset, DataLoader

# 1. env로부터 경로 가져오기
load_dotenv(dotenv_path="../../.env")

dataset_path = os.getenv("BLENDEDMVS_DATASET_PATH")

input_image_paths = sorted(glob.glob(os.path.join(dataset_path, "*", "blended_images", "*_masked.jpg")))
gt_image_paths = sorted(glob.glob(os.path.join(dataset_path, "*", "rendered_depth_maps", "*.pfm")))

# 변환 객체 생성 (이미지를 448x448로 센터크롭 후, [0,1] 범위의 텐서로 변환)
to_tensor = transforms.Compose([
    transforms.CenterCrop(448),
    transforms.ToTensor()
])


# input 이미지: RGB 이미지로 변환 후 텐서화
for path in input_image_paths:
    try:
        img = Image.open(path).convert('RGB')
        tensor_img = to_tensor(img)
        images.append(tensor_img)
        #print(f"Loaded image (tensor): {path}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

# gt (depth map): 'F' 모드로 변환하여 텐서화
for path in gt_image_paths:
    try:
        img = Image.open(path).convert('F')
        tensor_img = to_tensor(img)
        gt.append(tensor_img)
        #print(f"Loaded gt (tensor): {path}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

print(f"총 {len(images)}개의 이미지, {len(gt)}개의 정답")



load 


MVS_input = torch.stack(images)
MVS_gt = torch.stack(gt)



dataset = TensorDataset(MVS_input, MVS_gt)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


# DataLoader에서 첫 배치만 꺼내서 확인
for batch in dataloader:
    inputs, targets = batch
    print("Input batch shape:", inputs.shape)
    print("GT batch shape:", targets.shape)
    break  # 첫 배치만 확인