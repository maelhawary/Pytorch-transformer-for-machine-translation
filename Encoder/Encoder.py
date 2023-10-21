#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from Encoder import Encoder_block as Enco
import Dataset as dt
import config as config
from utils import Embedding as embd
from utils import Positional_Encoding as position

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self,src_vocab_size) -> None:
        super().__init__()
        self.confi=config.get_config()
        self.d_model=self.confi['d_model']
        self.seq_len= self.confi['seq_len'] 
        self.number_of_heads=self.confi['number_of_heads']
        self.number_of_layers=self.confi['number_of_layers']
        self.d_att_weigh=self.confi['d_att_weigh']
        self.d_FFN=self.confi['d_FFN']
        self.dropout=self.confi['dropout']
        self.vocab_size_src=src_vocab_size
        self.embd=embd.InputEmbeddings(self.d_model,self.vocab_size_src)   
        self.pos=position.PositionalEncoding(self.d_model,self.seq_len,self.dropout)
        self.blocks= Enco.EncoderBlock(self.number_of_heads,self.d_model,self.d_att_weigh,
                                                            self.seq_len,self.d_FFN,self.dropout) 
        self.LN=nn.LayerNorm(self.d_model)
        self.output = nn.Linear(self.d_model, self.vocab_size_src)
        self.device=device
    def print(self,idx_k,idx_q,idx_v,src_msk):
          print('num',src_msk)
       
    def forward(self,idx_k,idx_q,idx_v,src_msk):
        B,T=idx_k.shape
        out=self.embd(idx_k)
        out=self.pos(out) 
	
        for ii in range(self.number_of_layers):
          x=self.blocks(out,out,out,src_msk)
		
        Encoder_out=self.LN(x)  

        return Encoder_out




