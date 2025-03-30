import torch

print("CUDA :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU =:", torch.cuda.get_device_name(0))
else:
    print("GPU unavailable")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## 우리는 CPU inference 시를 제외하고는 cpu 쓸 생각 없으니, train에서는 cuda면 바로 에러 띄우기
