#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from Decoder import Decoder as decoder
from Encoder import Encoder as encoder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerEncoderDecoder(nn.Module):
	def __init__(self,src_vocab_size,tgt_vocab_size) -> None:
		super().__init__()                
		self.encoder=encoder.Encoder(src_vocab_size)
		self.decoder=decoder.Decoder(tgt_vocab_size)
        
	def forward(self,src,tgt,src_msk,tgt_msk):            
		endocer_output= self.encoder(src,src,src,src_msk)
		decoder_output=self.decoder(endocer_output,tgt,endocer_output,src_msk,tgt_msk)           
		return decoder_output




