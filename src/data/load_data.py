## 여기서 리스트 가져오는 함수를 하나 만들기. 
import os
import glob

def get_data_list(dataset_path,teacher=True):
    if teacher:
        input_image_paths = sorted(glob.glob(os.path.join(dataset_path, "*", "blended_images", "*_masked.jpg")))
        gt_image_paths = sorted(glob.glob(os.path.join(dataset_path, "*", "rendered_depth_maps", "*.pfm")))
    
    else :
        input_image_paths = sorted(glob.glob(os.path.join(dataset_path, "*", "blended_images", "*_masked.jpg")))