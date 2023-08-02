'''
help wav file read and write
This code have hyperparameters and classes
'''
from scipy.io import wavfile
import os
import yaml
import copy
import pickle

import librosa
import librosa.display

from pyworld import decode_spectral_envelope, synthesize

import numpy as np
import torch

import matplotlib.pyplot as plt

class hyperparams(object):
    def __init__(self):
        self.sr = 16000
        self.n_fft = 1024
        self.frame_shift = 0.0125 #s
        self.frame_length = 0.05 #s
        self.hop_length = int(self.sr*self.frame_shift)  # samples  This is dependent on the frame_shift.
        self.win_length = int(self.sr*self.frame_length)  # samples This is dependent on the frame_length.
        self.n_mels = 80 #number of Mel banks to generate
        