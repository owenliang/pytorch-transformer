'''
训练de翻译en模型
'''
from torch import nn 
import torch
from dataset import en_preprocess,de_preprocess,train_dataset,en_vocab,de_vocab,PAD_IDX
from transformer import Transformer
from torch.utils.data import DataLoader,Dataset
from config import DEVICE,SEQ_MAX_LEN
from torch.nn.utils.rnn import pad_sequence

# 数据集
class De2EnDataset(Dataset):
    def __init__(self):
        super().__init__()
        
        self.enc_x=[]
        self.dec_x=[]
        for de,en in train_dataset:
            # 分词
            de_tokens,de_ids=de_preprocess(de)
            en_tokens,en_ids=en_preprocess(en)
            # 序列超出的跳过
            if len(de_ids)>SEQ_MAX_LEN or len(en_ids)>SEQ_MAX_LEN:
                continue 
            self.enc_x.append(de_ids)
            self.dec_x.append(en_ids)
    
    def __len__(self):
        return len(self.enc_x)
    
    def __getitem__(self, index):
        return self.enc_x[index],self.dec_x[index]

def collate_fn(batch):
    enc_x_batch=[]
    dec_x_batch=[]
    for enc_x,dec_x in batch:
        enc_x_batch.append(torch.tensor(enc_x,dtype=torch.long))
        dec_x_batch.append(torch.tensor(dec_x,dtype=torch.long))

    # batch内序列长度补齐
    pad_enc_x=pad_sequence(enc_x_batch,True,PAD_IDX)
    pad_dec_x=pad_sequence(dec_x_batch,True,PAD_IDX)
    return pad_enc_x,pad_dec_x

if __name__=='__main__':
    # de翻译en的数据集
    dataset=De2EnDataset()
    dataloader=DataLoader(dataset,batch_size=250,shuffle=True,num_workers=4,persistent_workers=True,collate_fn=collate_fn)

    # 模型
    try:
        transformer=torch.load('checkpoints/model.pth')
    except:
        transformer=Transformer(enc_vocab_size=len(de_vocab),dec_vocab_size=len(en_vocab),emb_size=512,q_k_size=64,v_size=64,f_size=2048,head=8,nblocks=6,dropout=0.1,seq_max_len=SEQ_MAX_LEN).to(DEVICE)

    # 损失函数和优化器
    loss_fn=nn.CrossEntropyLoss(ignore_index=PAD_IDX) # 样本正确输出序列的pad词不参与损失计算
    optimizer=torch.optim.SGD(transformer.parameters(), lr=1e-3, momentum=0.99)

    # 开始练
    transformer.train()
    EPOCHS=300
    for epoch in range(EPOCHS):
        batch_i=0
        loss_sum=0
        for pad_enc_x,pad_dec_x in dataloader:
            real_dec_z=pad_dec_x[:,1:].to(DEVICE) # decoder正确输出
            pad_enc_x=pad_enc_x.to(DEVICE) 
            pad_dec_x=pad_dec_x[:,:-1].to(DEVICE) # decoder实际输入 
            dec_z=transformer(pad_enc_x,pad_dec_x) # decoder实际输出

            batch_i+=1
            loss=loss_fn(dec_z.view(-1,dec_z.size()[-1]),real_dec_z.view(-1)) # 把整个batch中的所有词拉平
            loss_sum+=loss.item()
            print('epoch:{} batch:{} loss:{}'.format(epoch,batch_i,loss.item()))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(transformer,'checkpoints/model.pth'.format(epoch))