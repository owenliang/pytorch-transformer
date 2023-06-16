import torch 

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')