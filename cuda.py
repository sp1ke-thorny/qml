import torch

print("GPU 是否可用:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("显卡型号:", torch.cuda.get_device_name(0))