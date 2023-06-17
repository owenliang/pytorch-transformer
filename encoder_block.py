'''
encoder block支持堆叠, 每个block都输入emb序列并输出emb序列(1:1对应)
'''
from torch import nn 
import torch 
from multihead_attn import MultiHeadAttention
from emb import EmbeddingWithPosition
from dataset import de_preprocess,train_dataset,de_vocab

class EncoderBlock(nn.Module):
    def __init__(self,emb_size,q_k_size,v_size,f_size,head):
        super().__init__()

        self.multihead_attn=MultiHeadAttention(emb_size,q_k_size,v_size,head)   # 多头注意力
        self.z_linear=nn.Linear(head*v_size,emb_size) # 调整多头输出尺寸为emb_size
        self.addnorm1=nn.LayerNorm(emb_size) # 按last dim做norm

        # feed-forward结构
        self.feedforward=nn.Sequential(
            nn.Linear(emb_size,f_size),
            nn.ReLU(),
            nn.Linear(f_size,emb_size)
        )
        self.addnorm2=nn.LayerNorm(emb_size) # 按last dim做norm

    def forward(self,x,attn_mask): # x: (batch_size,seq_len,emb_size)
        z=self.multihead_attn(x,x,attn_mask)  # z: (batch_size,seq_len,head*v_size)
        z=self.z_linear(z) # z: (batch_size,seq_len,emb_size)
        output1=self.addnorm1(z+x) # z: (batch_size,seq_len,emb_size)
        
        z=self.feedforward(output1) # z: (batch_size,seq_len,emb_size)
        return self.addnorm2(z+output1) # (batch_size,seq_len,emb_size)

if __name__=='__main__':
    # 准备1个batch
    emb=EmbeddingWithPosition(len(de_vocab),128)
    de_tokens,de_ids=de_preprocess(train_dataset[0][0]) # 取de句子转词ID序列
    de_ids_tensor=torch.tensor(de_ids,dtype=torch.long)
    emb_result=emb(de_ids_tensor.unsqueeze(0)) # 转batch再输入模型
    print('emb_result:', emb_result.size())

    attn_mask=torch.zeros((1,de_ids_tensor.size()[0],de_ids_tensor.size()[0])) # batch中每个样本对应1个注意力矩阵

    # 5个Encoder block堆叠
    encoder_blocks=[]
    for i in range(5):
        encoder_blocks.append(EncoderBlock(emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8))
    
    # 前向forward
    encoder_outputs=emb_result
    for i in range(5):
        encoder_outputs=encoder_blocks[i](encoder_outputs,attn_mask)
    print('encoder_outputs:',encoder_outputs.size())