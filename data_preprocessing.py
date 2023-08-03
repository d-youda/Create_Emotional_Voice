import torch
from utils.data_preprocess_utils import get_wav_and_labels
import argparse
import os
import pickle
import numpy as np
from shutil import copyfile
from utils.preprocess_world import world_features, cal_mcep, get_f0_stats


def copy_files(origin_data_dir, process_data_dir):
    '''copy할 directory만들기
    만든 directory에 오디오 파일들 넣고, annotation도 넣기'''
    process_audio_dir = os.path.join(process_data_dir, 'audio')
    process_annotations_dir = os.path.join(process_data_dir, 'annotation')

    if not os.path.exists(process_audio_dir):
        os.mkdir(process_audio_dir)
    if not os.path.exists(process_annotations_dir):
        os.mkdir(process_annotations_dir)
    
    for wav in os.listdir(origin_data_dir):
        annotation_dir = os.path.join(origin_data_dir,"label")
        for file in os.listdir(annotation_dir):
            if not file.endswith('.txt'):
                continue
            src_file = os.path.join(annotation_dir, file)
            dest_file = os.path.join(process_annotations_dir,file)
            if not os.path.exists(dest_file):
                copyfile(src_file, dest_file)
        
        # wav_dir = os.path.join(data_dir,"wav")
        for emotion in os.listdir(origin_data_dir):
            emotion_dir = os.path.join(origin_data_dir, emotion)
            for file in os.listdir(emotion):
                if not file.endswith(".wav"):
                    continue
                src_file = os.path.join(emotion,file)
                dest_file = os.path.join(process_audio_dir,file)
                if not os.path.exists(dest_file):
                    copyfile(src_file,dest_file)
        print(emotion+"completed.")

def generate_world_features(filenames, data_dir):
    '''create and save world features and labels'''

    world_dir = os.path.join(data_dir, 'world')
    f0_dir = os.path.join(data_dir, "f0")
    label_dir = os.path.join(data_dir, "labels")

    if not os.path.exists(world_dir):
        os.mkdir(world_dir)
    if not os.path.exists(f0_dir):
        os.mkdir(f0_dir)
    if not os.path.exists(label_dir):
        os.mkdir(label_dir)
    
    MIN_LENGTH = 0
    MAX_LENGTH = 1719
    worlds_made = 0

    for i, f in enumerate(filenames):
        wav, labels = get_wav_and_labels(f, data_dir)
        wav = np.array(labels)

        coded_sp_name = os.path.join(world_dir, f[:-4] + ".npy")
        label_name = os.path.join(label_dir, f[:-4] + ".npy")
        f0_name = os.path.join(f0_dir, f[:-4] + ".npy")
        if os.path.exists(coded_sp_name) and os.path.exists(label_name) and os.path.exists(f0_name):
            worlds_made += 1
            continue

        if labels[0] != -1:
            f0, ap, sp, coded_sp = cal_mcep(wav)

            if coded_sp.shape[1] <MAX_LENGTH:

                np.save(os.path.join(world_dir, f[:-4] + ".npy"), coded_sp)
                np.save(os.path.join(label_dir, f[:-4] + ".npy"), labels)
                np.save(os.path.join(f0_dir, f[:-4] + ".npy"), f0)

                worlds_made += 1
        
        if i % 10 == 0:
            print(i, " complete.")
            print(worlds_made, "worlds made.")

def generate_f0_stats(filenames, data_dir):
    NUM_SPEAKER = 10
    NUM_EMOTIONS = 5
    f0_dir = os.path.join(data_dir, 'f0')

    emo_stats = {}
    for e in range(NUM_EMOTIONS):
        spk_dict = {}
        for s in range(NUM_SPEAKER):
            f0s=[]
            for f in filenames:
                wav, labels = get_wav_and_labels(f, data_dir)
                wav = np.array(wav, dtype=np.float64)
                labels = np.array(labels)
                if labels[0] == e and labels[1] == s:
                    f0_file = os.path.join(f0_dir, f[:-4]+".npy")
                    if os.path.exists(f0_file):
                        f0 = np.load(f0_file)
                        f0s.append(f0)
            log_f0_mean, f0_std = get_f0_stats(f0s)
            spk_dict[s] = (log_f0_mean, f0_std)
            print(f"Done emotion{e}, speaker {s}.")
        emo_stats[e] = spk_dict
    
    with open('f0_dict.pkl', 'wb') as absolute_file:
        pickle.dump(emo_stats, absolute_file, pickle.HIGHEST_PROTOCOL)
    print("------ Absolute F0 stats Completed ------")

    for tag, val in emo_stats.items():
        print(f"Emotion {tag} stats: ")
        for tag2, val2 in val.items():
            print(f'{tag2} = {val2[0]}, {val2[1]}')
    
    #calculate relative F0 stats
    emo2emo_dict = {}
    for e1 in range(NUM_EMOTIONS):
        for e2 in range(NUM_EMOTIONS):
            mean_list = []
            std_list = []

            for s in range(NUM_SPEAKER):
                mean_diff = emo_stats[e2][s][0] - emo_stats[e1][s][0]
                std_diff = emo_stats[e2][s][1] - emo_stats[e1][s][1]
                mean_list.append(mean_diff)
                std_list.append(std_diff)
            
            mean_mean = np.mean(mean_list)
            std_mean = np.mean(std_list)
            emo2emo_dict[e1][e2] = (mean_mean, std_mean)
    
    print(" ---- Relative f0 stats completed ----")
    for tag, val in emo2emo_dict.items():
        print(f'Emotion {tag} stats:')
        for tag2, val2 in val2.items():
            print(f'{tag2} = {val2[0]}, {val2[1]}')
    with open('f0_relative_dict.pkl', 'wb') as relative_file:
        pickle.dump(emo2emo_dict, relative_file, pickle.HIGHEST_PROTOCOL)
        
def run_preprocessing(args):
    print(f"--------------------------{args.process_data_dir}의 Dataset 재구성 시작--------------------------")
    copy_files(args.origin_data_dir, args.process_data_dir)

    data_dir = args.process_data_dir
    audio_dir = os.path.join(data_dir, 'audio')
    audio_filenames = [f for f in os.listdir(audio_dir) if '.wav' in f]

    print("----------------- Producing WORLD features data -----------------")
    generate_world_features(audio_filenames, data_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main preprocessing pipeline')
    parser.add_argument("--origin_data_dir", type=str, help="Directory of voice wav file dataset")
    parser.add_argument("--process_data_dir", type=str, default='./processed_data',
                        help="Directory to copy audio and annotation files to.")

    args = parser.parse_args()

    run_preprocessing(args)