# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function

import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_slim
import os
import time
import Constants
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='vggish_model.ckpt', help='Path to the VGGish checkpoint file.')
parser.add_argument('--n_frames', type=int, default=28, help='the number of features per wav file. 0 means extrating non-overlapping features')
parser.add_argument('--dataset', type=str, default='MSRVTT')
parser.add_argument('--video_postfix', type=str, default='.mp4')
parser.add_argument('--n_latency_samples', type=int, default=1000)
args = parser.parse_args()

def main():
  base_path = os.path.join(Constants.base_data_path, args.dataset)
  wav_path = os.path.join(base_path, Constants.wav_folder_name)
  

  print('- Loading wav files from', wav_path)
  
  if args.dataset == 'MSVD':
    args.video_postfix = '.avi'

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, args.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    files = os.listdir(wav_path)[:args.n_latency_samples]
    total_time = 0

    for wav_file in tqdm(files):
        vid = wav_file.split('.')[0]
        
        examples_batch = vggish_input.wavfile_to_examples2(
          os.path.join(wav_path, wav_file),  
          base_path, 
          n_frames=int(args.n_frames), 
          video_postfix=args.video_postfix
        )

        start_time = time.time()
        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        
        total_time += (time.time() - start_time)
    
    print(f'- # samples: {len(files)}')
    print(f'- Total inference time: {total_time}')
    print(f'- Average latency: {total_time / len(files)}')
      
    with open('latency.txt', 'a') as f:
      f.write(f'{len(files)}\t{total_time}\t{total_time / len(files)}\n')

if __name__ == '__main__':
  main()
