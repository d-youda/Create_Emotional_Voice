'''
classifier_emotion.py

Author - yudahyeon

An Improved StarGAN for Emotional Voice Conversion:
Enhancing Voice Quality and Data Augmentation code 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from StarGAN.stargan_attention import Attention

class Emotion_Classifier(nn.Module):
    def init(self,input_size, hidden_size, num_layers, bi=True, device='cuda'):
        super(Emotion_Classifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.m_factor = 2 if bi else 1
        self.num_classes = 5
        
        
        kernal = 7
        padding = 3
        self.conv1 = nn.Conv2d(in_channels=36, out_channels=16, kernel_size=kernal, stride=1, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=kernal, stride=1, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2, stride=1)
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=32, kernel_size=kernal, stride=1, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2, stride=1)

        self.bilstm1 = nn.LSTM(input_size=32, hidden_size=128, bidirectional=True)
        # self.bilstm2 = nn.LSTM(intput_size=)
        self.attention = Attention(self.hidden_size*self.m_factor)


        self.fc = nn.Linear(self.m_factor*hidden_size, 64)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(64, self.num_classes)
    
    def forward(self, x_data, x_lens):
        '''
        x.size = (batch_size, 1, max_seq_length, feature_dim)
        x_lens is size(batch_size, 1), seq_lens포함
        '''
        batch_size = x_data.size(0)
        no_features = x_data.suze(3)
        curr_device = x_data.device

        #Conv layer
        x_data = self.maxpool1(F.relu(self.conv1(x_data)))
        x_data = self.maxpool2(F.relu(self.conv2(x_data)))
        x_data = self.maxpool3(F.relu(self.conv3(x_data)))
        x_lens = x_lens//8 # x = (B, channels, max_l//4, n_mels//4)

        #RNN layer
        x_data = x_data.permute(0,2,1,3)
        x_data = x_data.contiguous().view(batch_size, -1, 32*(no_features//8))

        x_data = nn.utils.rnn.pack_padded_sequence(x_data, x_lens,
                                                   batch_first=True, 
                                                   enforce_sorted=True)
        h0 = torch.zeros(self.m_factor*32, batch_size,self.hidden_size).to(device=curr_device,dtype=torch.float)
        c0 = torch.zeros(self.m_factor*32, batch_size,self.hidden_size).to(device=curr_device,dtype=torch.float)

        #LSTM return값 : (seq_len, batch, num_directions*hidden_size)
        x_data, _ = self.bilstm1(x_data, (h0,c0))
        x_data, x_lens = torch.nn.util.rnn.pad_packed_sequence(x_data, batch_first=True)

        x_data = self.att(x_data)

        x_data = self.drop(F.relu(self.fc(x_data)))

        x_data = self.out(x_data)

        return x_data
