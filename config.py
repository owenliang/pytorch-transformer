import torch 

# 设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 最长序列（受限于postition emb）
SEQ_MAX_LEN=5000