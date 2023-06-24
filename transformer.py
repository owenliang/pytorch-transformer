'''
transformer模型
由encoder和decoder构成
'''
from torch import nn 
import torch
from decoder import Decoder
from encoder import Encoder
from dataset import en_preprocess,de_preprocess,train_dataset,en_vocab,de_vocab,PAD_IDX
from config import DEVICE

class Transformer(nn.Module):
    def __init__(self,enc_vocab_size,dec_vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout=0.1,seq_max_len=5000):
        super().__init__()
        self.encoder=Encoder(enc_vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout,seq_max_len)
        self.decoder=Decoder(dec_vocab_size,emb_size,q_k_size,v_size,f_size,head,nblocks,dropout,seq_max_len)

    def forward(self,encoder_x,decoder_x):
        encoder_z=self.encode(encoder_x)
        return self.decode(decoder_x,encoder_z,encoder_x)

    def encode(self,encoder_x):
        encoder_z=self.encoder(encoder_x)
        return encoder_z

    def decode(self,decoder_x,encoder_z,encoder_x):
        decoder_z=self.decoder(decoder_x,encoder_z,encoder_x)
        return decoder_z
    
if __name__=='__main__':
    transformer=Transformer(enc_vocab_size=len(en_vocab),dec_vocab_size=len(de_vocab),emb_size=128,q_k_size=256,v_size=512,f_size=512,head=8,nblocks=3,dropout=0.1,seq_max_len=5000).to(DEVICE)
    
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

    # 输出每个en词的下一个词概率
    decoder_z=transformer(enc_x_batch,dec_x_batch)
    print('decoder outputs:',decoder_z.size())