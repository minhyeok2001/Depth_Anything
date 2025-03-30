import torch

print("CUDA :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU =:", torch.cuda.get_device_name(0))
else:
    print("GPU unavailable")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
