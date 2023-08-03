import pyworld
from utils import audio_utils
import numpy as np

FEATURE_DIM = 36
SAMPLE_RATE = 16000
FRAMES = 512
FFTSIZE = 1024

def world_features(wav, sr, fft_size, dim):
    f0, timeaxis = pyworld.harvest(wav, sr)
    sp = pyworld.cheaptrick(wav, f0, timeaxis, sr, fft_size)
    ap = pyworld.d4c(wav, f0, timeaxis, sr, fft_size)
    coded_sp = pyworld.code_spectral_envelope(sp, sr, dim)

    return f0, timeaxis, sp, ap, coded_sp

def cal_mcep(wav, sr=SAMPLE_RATE, dim=FEATURE_DIM, fft_size=FFTSIZE):
    '''given wav signal'''
    f0, timeaxis, sp, ap, coded_sp = world_features(wav, sr, fft_size, dim)

    if audio_utils.hp.normalise:
        coded_sp = audio_utils._normalise_coded_sp(coded_sp)
    
    coded_sp = coded_sp.T
    return f0, ap, sp, coded_sp

def get_f0_stats(f0s):
    log_f0s_concatenated = np.ma.log(np.concatenate(f0s))
    log_f0s_mean = log_f0s_concatenated.mean()
    log_f0s_std = np.var(log_f0s_concatenated)

    return log_f0s_mean, log_f0s_std