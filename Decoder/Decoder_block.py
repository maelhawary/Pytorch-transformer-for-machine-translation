#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from utils import Position_wise_FFN as FFN
from utils import Multi_attention_heads as Matt


class DecoderBlock(nn.Module):

    def __init__(self,number_of_heads,d_model,d_att_weigh,seq_len,d_FFN,dropout) -> None:
        super().__init__()

        self.self_att= Matt.Multiheads(number_of_heads, d_model, d_att_weigh, seq_len, dropout)
        self.LN_1=nn.LayerNorm(d_model)
        self.LN_2=nn.LayerNorm(d_att_weigh)
        self.FFN=FFN.PositionWiseFFN(d_att_weigh, d_FFN,dropout)
        self.cross_att= Matt.Multiheads(number_of_heads, d_model, d_att_weigh, seq_len, dropout)

    def forward(self,k,q,v,src_msk,tgt_msk):
        #print('shapp',idx.shape)
       # print('check-k-q',torch.max(k-q))
        out=q+self.self_att(self.LN_1(q),self.LN_1(q),self.LN_1(q),tgt_msk)
        out=out+self.cross_att(self.LN_1(k),self.LN_1(out),self.LN_1(v),src_msk)  
        out=out+self.FFN(self.LN_2(out))# (B,T,d_FFN)
        return out
    
    

