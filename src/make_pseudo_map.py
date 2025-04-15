import os
import glob
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from src.utils.check_device import device
from src.models.model import DepthModel
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
dataset_path = os.getenv("GOOGLELANDMARK_DATASET_PATH")

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} from {checkpoint_path}")
    return model

class PseudoImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.transform = transforms.Compose([
            transforms.CenterCrop(448),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, os.path.basename(image_path)

def generate_pseudo_labels(image_dir, save_dir, checkpoint_path, batch_size=4, num_workers=4):
    os.makedirs(save_dir, exist_ok=True)
    dataset = PseudoImageDataset(image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Found {len(dataset)} images in {image_dir}.")
    model = DepthModel(features=256, out_channels=[256, 512, 1024, 1024],
                       use_bn=False, localhub=False).to(device)
    model = load_checkpoint(model, checkpoint_path)
    model.eval()
    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            for i, fname in enumerate(filenames):
                save_path = os.path.join(save_dir, os.path.splitext(fname)[0] + ".npy")
                np.save(save_path, outputs[i])
                print(f"Saved pseudo label to {save_path}")

def main():
    checkpoint_path = "best_checkpoint_teacher.pth"
    train_img_dir = os.path.join(dataset_path, "image")
    train_save_dir = os.path.join(dataset_path, "pseudo_depth")
    generate_pseudo_labels(train_img_dir, train_save_dir, checkpoint_path)
    val_img_dir = os.path.join(dataset_path, "val", "image")
    val_save_dir = os.path.join(dataset_path, "val", "pseudo_depth")
    generate_pseudo_labels(val_img_dir, val_save_dir, checkpoint_path)

if __name__ == "__main__":
    main()