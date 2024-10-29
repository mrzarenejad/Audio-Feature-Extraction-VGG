from subprocess import call
import os
from tqdm import tqdm
import Constants
import argparse


def run(video_path, wav_path):
    os.makedirs(wav_path, exist_ok=True)
    for fn in tqdm(os.listdir(video_path)):
        vid = fn.split('.')[0]
        video_file = os.path.join(video_path, fn)
        wav_out = os.path.join(wav_path, '%s.wav'%vid)
        if os.path.exists(wav_out): continue
        call(["ffmpeg", "-i", video_file, "-f", "wav", wav_out])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MSRVTT')
    args = parser.parse_args()
    
    video_path = os.path.join(Constants.base_data_path, args.dataset, Constants.video_folder_name)
    wav_path = os.path.join(Constants.base_data_path, args.dataset, Constants.wav_folder_name)

    print('- Loading video files from', video_path)
    print('- Saving wav files to', wav_path)

    run(video_path, wav_path)
