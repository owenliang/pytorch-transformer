'''
encoder编码器,输入词id序列,输出每个词的编码向量(输入输出1:1)
'''
from torch import nn 
import torch 
from encoder_block import EncoderBlock
from emb import EmbeddingWithPosition
from dataset import de_preprocess,train_dataset,de_vocab,PAD_IDX
from config import DEVICE

class Encoder(nn.Module):
    def __init__(self,vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout=0.1,seq_max_len=5000):
        super().__init__()
        self.emb=EmbeddingWithPosition(vocab_size,emb_size,dropout,seq_max_len)

        self.encoder_blocks=nn.ModuleList()
        for _ in range(nblocks):
            self.encoder_blocks.append(EncoderBlock(emb_size,q_k_size,v_size,f_size,head))

    def forward(self,x): # x:(batch_size,seq_len)
        pad_mask=(x==PAD_IDX).unsqueeze(1) # pad_mask:(batch_size,1,seq_len)
        pad_mask=pad_mask.expand(x.size()[0],x.size()[1],x.size()[1]) # pad_mask:(batch_size,seq_len,seq_len)

        pad_mask=pad_mask.to(DEVICE)

        x=self.emb(x)
        for block in self.encoder_blocks:
            x=block(x,pad_mask) # x:(batch_size,seq_len,emb_size)
        return x
    
if __name__=='__main__':
    # 取2个de句子转词ID序列
    de_tokens1,de_ids1=de_preprocess(train_dataset[0][0]) 
    de_tokens2,de_ids2=de_preprocess(train_dataset[1][0]) 

    # 组成batch并padding对齐
    if len(de_ids1)<len(de_ids2):
        de_ids1.extend([PAD_IDX]*(len(de_ids2)-len(de_ids1)))
    elif len(de_ids1)>len(de_ids2):
        de_ids2.extend([PAD_IDX]*(len(de_ids1)-len(de_ids2)))
    
    batch=torch.tensor([de_ids1,de_ids2],dtype=torch.long).to(DEVICE)
    print('batch:', batch.size())

    # Encoder编码
    encoder=Encoder(vocab_size=len(de_vocab),emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8,nblocks=3).to(DEVICE)
    z=encoder.forward(batch)
    print('encoder outputs:', z.size())