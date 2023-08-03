import librosa
import os
from shutil import copy2
data_path = '.\Dataset'
audio_path = os.path.join(data_path,'audio')
label_path = os.path.join(data_path,"annotation")
if not os.path.exists(label_path):
    os.mkdir(label_path)

audio_data_path = './Data/wav'
if not os.path.exists(audio_data_path):
    os.mkdir(audio_data_path)

for emotions in os.listdir(audio_path):
    emotion_path = os.path.join(audio_path,emotions)
    for audio in os.listdir(emotion_path):
        filename = audio
        emotion = emotions
        number_of_dim = 36 #추후 수정
        anno = open(f'{label_path/audio}.txt', 'w+')
        anno.write('[filename\temotion\tnumber_of_dim\t]')
        
        src_file = os.path.join(emotion_path,audio)
        dest_dir = audio_data_path
        copy2(src_file, dest_dir)
