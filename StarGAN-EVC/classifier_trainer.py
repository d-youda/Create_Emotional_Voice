import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import yaml
import argparse
import librosa

from utils import audio_utils
import StarGAN.my_dataset as my_dataset
import StarGAN.classifier_emotion as classifiers
from StarGAN.my_dataset import get_filenames
from train_main import make_weight_vector

import torchvision
import sklearn
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print("Device used: ", device)

def save_checkpoint(state, filename='./checkpoint/classifier.ckpt')
    
    print("Saving a new best model!")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(state, filename)

def load_checkpoints(model, optimiser, filename='./checkpoint/classifier.ckpt'):
    
    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    epoch = checkpoint['epoch']

    return epoch

def train_model(model, optimiser, train_data_loader, val_data_loader, loss_fn,
                model_type='cls', epochs=1, print_every=1, var_len_data=False, start_epoch=1):
    model = model.to(device=device)

    print("Training model type : ", model_type)
    best_model_score=0

    for e in range(start_epoch, epochs+1):
        total_loss = 0
        for t,(x,y) in enumerate(train_data_loader):
            model.train()

            if(var_len_data):
                x_real = x[0].to(device=device).unsqueeze(1)
                x_lens = x[1].to(device=device)
            else:
                x_real = x.to(device=device, dtype=torch.float)


if __name__ =='__main__':
    parser