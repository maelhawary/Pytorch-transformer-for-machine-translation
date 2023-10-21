import torch 
import torch.nn as nn

# A two layer feedforward neural network that helps in mitigating the bias caused by the self-attention model

class PositionWiseFFN(nn.Module):

    def __init__(self,d_att_weigh,d_FFN,dropout) -> None:
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(d_att_weigh, d_FFN),
            nn.ReLU(),
            nn.Linear(d_FFN, d_att_weigh),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)        