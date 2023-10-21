

import Dataset as dt
import torch
import torch.nn as nn
from config import get_config
from Transformer_block import Transformer_encoder_decoder as Transfor
import os
import tqdm as tqdm


def train(device,config,dir):
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = dt.get_ds(config)
    model = Transfor.TransformerEncoderDecoder(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
    # intialize the model weights before training    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    initial_epoch = 0
    global_step = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()      
        for batch in train_dataloader:
            encoder_input = batch['encoder_input'].to(device) 
            decoder_input = batch['decoder_input'].to(device) 
            encoder_mask = batch['encoder_mask'].to(device) 
            decoder_mask = batch['decoder_mask'].to(device) 
            
            pred = model(encoder_input,decoder_input,encoder_mask,decoder_mask) 
            label = batch['label'].to(device) 
            loss = loss_fn(pred.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            print(f"Epoch {epoch+1} Loss: {loss:.4f} Global_iter: {global_step}")
        
            if not os.path.exists(dir):
                os.makedirs(dir) 
        print(f"Epoch {epoch+1} Loss: {loss:.4f} Global_iter: {global_step}")
        torch.save(model , dir+'model_iter_'+str(epoch)+'.pth')
        torch.save(model.state_dict(), dir+'mode_state_iter_'+str(epoch)+'.pt') 
        torch.save(optimizer.state_dict(), dir+'optimizer_state_dict_'+str(epoch)+'.pt')  


                 