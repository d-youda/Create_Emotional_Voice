import torch
from utils import audio_utils
import argparse
import os

def copy_files(origin_data_dir, process_data_dir):
    '''copy할 directory만들기
    만든 directory에 오디오 파일들 넣고, annotation도 넣기'''
    prrocess_audio_dir = os.path.join(process_data_dir, 'audio')
    process_annotations_dir = os.path.join(process_data_dir, 'annotation')

    if not os.path.exists(prrocess_audio_dir):
        os.mkdir(prrocess_audio_dir)
    if not os.path.exists(process_annotations_dir):
        os.mkdir(process_annotations_dir)
    
    for wav in os.listdir(origin_data_dir):
        
    
def run_preprocessing(args):
    print(f"--------------------------{args.process_data_dir}의 Dataset 재구성 시작--------------------------")
    copy_files(args.origin_data_dir, args.process_data_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Main preprocessing pipeline')
    parser.add_argument("--origin_data_dir", type=str, help="Directory of IEMOCAP dataset")
    parser.add_argument("--process_data_dir", type=str, default='./processed_data',
                        help="Directory to copy audio and annotation files to.")

    args = parser.parse_args()

    run_preprocessing(args)