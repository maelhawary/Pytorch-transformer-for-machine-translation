import train as tr
from config import get_config
import torch

if __name__ == '__main__':
    # specify the direction to save the traning model
	dir='save_models'+'/'
    # loading the device
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	print("Using device:", device)
    # load the configuration file for hyperparameters and model specifications
	config=get_config()
    # train the mdoel
	tr.train(device,config,dir)