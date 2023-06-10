'''
输入emb后的词序列,根据Q,K,V方法计算词与词之间的相关性,为每个词生成信息提取后的emb(与输入词1:1映射)
'''
from torch import nn 
import torch 
from dataset import de_vocab,de_preprocess,train_dataset
from emb import EmbeddingWithPosition

class MultiHeadAttention(nn.Module):
    def __init__(self,emb_size,q_k_size,v_size,head):
        super().__init__()
        self.w_q=nn.Linear(emb_size,q_k_size)
        self.w_k=nn.Linear(emb_size,q_k_size)
        self.w_v=nn.Linear(emb_size,v_size)

    def forward(self,x_q,x_k_v,attn_mask):
        # x_q: (batch_size,seq_len,emb_size)
        q=self.w_q(x_q) # q: (batch_size,seq_len,q_k_size)
        k=self.w_k(x_k_v) # k: (batch_size,seq_len,q_k_size)
        attn=torch.matmul(q,k.transpose(1,2)) # (batch_size,seq_len,seq_len) ,  注意力矩阵,row是q,col是k
        
        # todo: mask
        
        # todo: head
        v=self.w_v(x_k_v) # v: (batch_size,seq_len,v_size)
        z=torch.matmul(attn,v) # z: (batch_size,seq_len,v_size), 多头输出
        return z

if __name__=='__main__':
    # 准备1个batch
    emb=EmbeddingWithPosition(len(de_vocab),128)
    de_tokens,de_ids=de_preprocess(train_dataset[0][0]) # 取de句子转词ID序列
    de_ids_tensor=torch.tensor(de_ids,dtype=torch.long)
    emb_result=emb(de_ids_tensor.unsqueeze(0)) # 转batch再输入模型
    print('emb_result:', emb_result.size())

    # 多头注意力
    multihead=MultiHeadAttention(emb_size=128,q_k_size=256,v_size=512,head=8)
    multihead_result=multihead.forward(x_q=emb_result,x_k_v=emb_result,attn_mask=None)
    print('multihead_result:', multihead_result.size())