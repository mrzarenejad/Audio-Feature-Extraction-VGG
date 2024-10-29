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

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_slim

import h5py
import Constants
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='D:/Users/zare/DeepLearning/VideoCaption/Dataset/EMVIDCAP/EmVidCap/Videos/EmVidCap-C/code/vggish-master/vggish_model.ckpt', help='Path to the VGGish checkpoint file.')
parser.add_argument('--n_frames', type=int, default=Constants.n_total_frames, 
    help='the number of features per wav file. 0 means extrating non-overlapping features',
    choices=[0, Constants.n_total_frames])
parser.add_argument('--dataset', type=str, default='MSRVTT')
parser.add_argument('--video_postfix', type=str, default='.mp4')
args = parser.parse_args()

def main():
  base_path = os.path.join(Constants.base_data_path, args.dataset)
  wav_path = os.path.join(base_path, Constants.wav_folder_name)
  save_path = os.path.join(base_path, 'feats')
  os.makedirs(save_path, exist_ok=True)
  if args.n_frames == 0:
    save_name = 'audio_vggish_audioset_overlap0.hdf5'
  else:
    save_name = 'audio_vggish_audioset_fixed%d.hdf5' % args.n_frames
  save_path = os.path.join(save_path, save_name)

  print('- Loading wav files from', wav_path)
  print('- Saving features to', save_path)
  
  if args.dataset == 'MSVD':
    args.video_postfix = '.avi'

  db = h5py.File(save_path, 'a')

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, args.checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    for wav_file in tqdm(os.listdir(wav_path)):
        vid = wav_file.split('.')[0]
        
        if args.dataset == 'MSRVTT' and int(vid[5:]) >= 10000:
          # this is because MSR-VTT (2017) has 13000 videos, we use MSR-VTT (2016).
          continue

        if vid in db.keys():
          continue

        examples_batch = vggish_input.wavfile_to_examples2(
          os.path.join(wav_path, wav_file),  
          base_path, 
          n_frames=int(args.n_frames), 
          video_postfix=args.video_postfix
        )

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        db[vid] = embedding_batch

  db.close()


if __name__ == '__main__':
  main()
