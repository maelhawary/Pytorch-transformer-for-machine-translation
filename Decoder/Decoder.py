#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from Decoder import Decoder_block as Deco
import Dataset as dt
import config as config
from utils import Embedding as embd
from utils import Positional_Encoding as position

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Decoder(nn.Module):
    def __init__(self,tgt_vocab_size) -> None:
        super().__init__()
        self.confi=config.get_config()
        self.d_model=self.confi['d_model']
        self.seq_len= self.confi['seq_len'] 
        self.number_of_heads=self.confi['number_of_heads']
        self.number_of_layers=self.confi['number_of_layers']
        self.d_att_weigh=self.confi['d_att_weigh']
        self.d_FFN=self.confi['d_FFN']
        self.dropout=self.confi['dropout']
        self.vocab_size_tgt=tgt_vocab_size    
        self.embd=embd.InputEmbeddings(self.d_model,self.vocab_size_tgt)   
        self.pos=position.PositionalEncoding(self.d_model,self.seq_len,self.dropout)
        self.blocks_decoder= Deco.DecoderBlock(self.number_of_heads,self.d_model,self.d_att_weigh,
                                                            self.seq_len,self.d_FFN,self.dropout)
        self.LN=nn.LayerNorm(self.d_model)
        self.output = nn.Linear(self.d_model, self.vocab_size_tgt)
        self.device=device
        

    def forward(self,idx_k,idx_q,idx_v,src_msk,tgt_msk):
                
        B,T=idx_q.shape 
        x=self.embd(idx_q)
        x=self.pos(x) 
       
        for ii in range(self.number_of_layers):
          x=self.blocks_decoder(idx_k,x,idx_v,src_msk,tgt_msk)

        x=self.LN(x)
        pred=self.output(x)

        return pred




