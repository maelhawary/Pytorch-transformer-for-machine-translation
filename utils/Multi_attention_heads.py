import torch
import torch.nn as nn



class AttentionHead(nn.Module):

    def __init__(self,d_model,seq_len,head_size,dropout) -> None:
        super().__init__()       
        self.K=nn.Linear(d_model,head_size, bias=False)
        self.Q=nn.Linear(d_model,head_size, bias=False)
        self.V=nn.Linear(d_model,head_size, bias=False)       
        self.dropout = nn.Dropout(dropout)

    def forward(self,idx_k,idx_q,idx_v, msk):
        B,L,d_model = idx_k.shape
        Key=self.K(idx_k) 
        Query=self.Q(idx_q) 
        Value=self.V(idx_v) 
        S=Query @ Key.transpose(-2,-1) * Key.shape[-1]**-0.5
        S_masked=S.masked_fill(msk == 0, -1e9)
        att=nn.functional.softmax(S_masked, dim=-1)
        att = self.dropout(att)
        out= att @ Value
        #return out,att
        return out

# here we split the attention head into multi-heads that are trained in parallel to help in the exploration
# process in the training   
class Multiheads(nn.Module):
    def __init__(self,number_of_heads,d_model,d_att_weigh,seq_len,dropout) -> None:
        super().__init__()

        assert d_att_weigh % number_of_heads == 0, "d_att_weigh is not divisible by number_of_heads"
        self.head_size=d_att_weigh // number_of_heads
        self.Multiheads=nn.ModuleList([AttentionHead(d_model,seq_len,self.head_size,dropout) for i in range (number_of_heads) ])#
        self.proj = nn.Linear(self.head_size * number_of_heads, d_att_weigh)
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx_k,idx_q,idx_v, src_msk):
        out = torch.cat([h(idx_k,idx_q,idx_v, src_msk) for h in self.Multiheads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
