'''
decoder解码器, 输出当前词序列的下一个词概率
'''
from torch import nn 
import torch 
from emb import EmbeddingWithPosition
from dataset import de_preprocess,en_preprocess,train_dataset,de_vocab,PAD_IDX,en_vocab
from decoder_block import DecoderBlock
from encoder import Encoder
from config import DEVICE

class Decoder(nn.Module):
    def __init__(self,vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout=0.1,seq_max_len=5000):
        super().__init__()
        self.emb=EmbeddingWithPosition(vocab_size,emb_size,dropout,seq_max_len)

        self.decoder_blocks=nn.ModuleList()
        for _ in range(nblocks):
            self.decoder_blocks.append(DecoderBlock(emb_size,q_k_size,v_size,f_size,head))
        
        # 输出向量词概率Logits
        self.linear=nn.Linear(emb_size,vocab_size)  

    def forward(self,x,encoder_z,encoder_x): # x: (batch_size,seq_len)
        first_attn_mask=(x==PAD_IDX).unsqueeze(1).expand(x.size()[0],x.size()[1],x.size()[1]).to(DEVICE) # 目标序列的pad掩码
        first_attn_mask=first_attn_mask|torch.triu(torch.ones(x.size()[1],x.size()[1]),diagonal=1).bool().unsqueeze(0).expand(x.size()[0],-1,-1).to(DEVICE) # &目标序列的向后看掩码
        # 根据来源序列的pad掩码，遮盖decoder对其pad部分的注意力
        second_attn_mask=(encoder_x==PAD_IDX).unsqueeze(1).expand(encoder_x.size()[0],x.size()[1],encoder_x.size()[1]).to(DEVICE) # (batch_size,target_len,src_len)

        x=self.emb(x)
        for block in self.decoder_blocks:
            x=block(x,encoder_z,first_attn_mask,second_attn_mask)
        
        return self.linear(x) # (batch_size,target_len,vocab_size)
    
if __name__=='__main__':
    # 取2个de句子转词ID序列，输入给encoder
    de_tokens1,de_ids1=de_preprocess(train_dataset[0][0]) 
    de_tokens2,de_ids2=de_preprocess(train_dataset[1][0]) 
    # 对应2个en句子转词ID序列，再做embedding，输入给decoder
    en_tokens1,en_ids1=en_preprocess(train_dataset[0][1]) 
    en_tokens2,en_ids2=en_preprocess(train_dataset[1][1])

    # de句子组成batch并padding对齐
    if len(de_ids1)<len(de_ids2):
        de_ids1.extend([PAD_IDX]*(len(de_ids2)-len(de_ids1)))
    elif len(de_ids1)>len(de_ids2):
        de_ids2.extend([PAD_IDX]*(len(de_ids1)-len(de_ids2)))
    
    enc_x_batch=torch.tensor([de_ids1,de_ids2],dtype=torch.long).to(DEVICE)
    print('enc_x_batch batch:', enc_x_batch.size())

    # en句子组成batch并padding对齐
    if len(en_ids1)<len(en_ids2):
        en_ids1.extend([PAD_IDX]*(len(en_ids2)-len(en_ids1)))
    elif len(en_ids1)>len(en_ids2):
        en_ids2.extend([PAD_IDX]*(len(en_ids1)-len(en_ids2)))
    
    dec_x_batch=torch.tensor([en_ids1,en_ids2],dtype=torch.long).to(DEVICE)
    print('dec_x_batch batch:', dec_x_batch.size())

    # Encoder编码,输出每个词的编码向量
    enc=Encoder(vocab_size=len(de_vocab),emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8,nblocks=3).to(DEVICE)
    enc_outputs=enc(enc_x_batch)
    print('encoder outputs:', enc_outputs.size())

    # Decoder编码,输出每个词对应下一个词的概率
    dec=Decoder(vocab_size=len(en_vocab),emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8,nblocks=3).to(DEVICE)
    enc_outputs=dec(dec_x_batch,enc_outputs,enc_x_batch)
    print('decoder outputs:', enc_outputs.size())