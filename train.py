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

r"""A simple demonstration of running VGGish in training mode.

This is intended as a toy example that demonstrates how to use the VGGish model
definition within a larger model that adds more layers on top, and then train
the larger model. If you let VGGish train as well, then this allows you to
fine-tune the VGGish model parameters for your application. If you don't let
VGGish train, then you use VGGish as a feature extractor for the layers above
it.

For this toy task, we are training a classifier to distinguish between three
classes: sine waves, constant signals, and white noise. We generate synthetic
waveforms from each of these classes, convert into shuffled batches of log mel
spectrogram examples with associated labels, and feed the batches into a model
that includes VGGish at the bottom and a couple of additional layers on top. We
also plumb in labels that are associated with the examples, which feed a label
loss used for training.

Usage:
  # Run training for 100 steps using a model checkpoint in the default
  # location (vggish_model.ckpt in the current directory). Allow VGGish
  # to get fine-tuned.
  $ python vggish_train_demo.py --num_batches 100

  # Same as before but run for fewer steps and don't change VGGish parameters
  # and use a checkpoint in a different location
  $ python vggish_train_demo.py --num_batches 50 \
                                --train_vggish=False \
                                --checkpoint /path/to/model/checkpoint
"""

from __future__ import print_function

from random import shuffle

import numpy as np
import tensorflow.compat.v1 as tf
import tf_slim as slim

import vggish_input
import vggish_params
import vggish_slim

flags = tf.app.flags

flags.DEFINE_integer(
    'num_epochs', 3,
    '.')

flags.DEFINE_integer(
    'batch_size', 128,
    '.')

flags.DEFINE_integer(
    'log_step', 300,
    '.')

flags.DEFINE_boolean(
    'train_vggish', True,
    'If True, allow VGGish parameters to change during training, thus '
    'fine-tuning VGGish. If False, VGGish parameters are fixed, thus using '
    'VGGish as a fixed feature extractor.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

FLAGS = flags.FLAGS

_NUM_CLASSES = 20


import os
from tqdm import tqdm
import pickle
import h5py

def _save_all_examples(wav_file_path, itoc):
  db = h5py.File('/home/yangbang/VC_data/MSRVTT/feats/vggish_examples.hdf5', 'a')
  for fn in tqdm(os.listdir(wav_file_path)):
    _id = int(fn.split('.')[0])
    if str(_id) in db.keys(): continue
    examples = vggish_input.wavfile_to_examples(os.path.join(wav_file_path, fn))
    db[str(_id)] = examples
  db.close()

def _get_all_examples(wav_file_path, itoc):
  _save_all_examples(wav_file_path, itoc)
  db = h5py.File('/home/yangbang/VC_data/MSRVTT/feats/vggish_examples.hdf5', 'r')
  all_examples = []
  all_labels = []
  for _id in db.keys():
    examples = np.asarray(db[_id])
    category = [0] * 20
    category[itoc[int(_id)]] = 1
    labels = np.array([category] * examples.shape[0])
    all_examples.append(examples)
    all_labels.append(labels)

  all_examples = np.concatenate(all_examples)
  all_labels = np.concatenate(all_labels)
  db.close()
  return all_examples, all_labels


def main(_):
  all_examples, all_labels = _get_all_examples('/home/yangbang/VC_data/MSRVTT/all_wavs', pickle.load(open('/home/yangbang/VC_data/MSRVTT/info_corpus.pkl', 'rb'))['info']['itoc'])
  with tf.Graph().as_default(), tf.Session() as sess:
    # Define VGGish.
    embeddings = vggish_slim.define_vggish_slim(training=FLAGS.train_vggish)

    # Define a shallow classification model and associated training ops on top
    # of VGGish.
    with tf.variable_scope('mymodel'):
      # Add a fully connected layer with 100 units. Add an activation function
      # to the embeddings since they are pre-activation.
      num_units = 100
      fc = slim.fully_connected(tf.nn.relu(embeddings), num_units)

      # Add a classifier layer at the end, consisting of parallel logistic
      # classifiers, one per class. This allows for multi-class tasks.
      logits = slim.fully_connected(
          fc, _NUM_CLASSES, activation_fn=None, scope='logits')
      tf.sigmoid(logits, name='prediction')

      # Add training ops.
      with tf.variable_scope('train'):
        global_step = tf.train.create_global_step()

        # Labels are assumed to be fed as a batch multi-hot vectors, with
        # a 1 in the position of each positive class label, and 0 elsewhere.
        labels_input = tf.placeholder(
            tf.float32, shape=(None, _NUM_CLASSES), name='labels')

        # Cross-entropy label loss.
        xent = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels_input, name='xent')
        loss = tf.reduce_mean(xent, name='loss_op')
        tf.summary.scalar('loss', loss)

        # We use the same optimizer and hyperparameters as used to train VGGish.
        optimizer = tf.train.AdamOptimizer(
            learning_rate=1e-5,
            epsilon=vggish_params.ADAM_EPSILON)
        train_op = optimizer.minimize(loss, global_step=global_step)

    # Initialize all variables in the model, and then load the pre-trained
    # VGGish checkpoint.
    sess.run(tf.global_variables_initializer())
    vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

    # The training loop.
    features_input = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    
    total_loss = 0
    saver = tf.train.Saver()
    for _ in tqdm(range(FLAGS.num_epochs)):
      labeled_examples = list(zip(all_examples, all_labels))
      shuffle(labeled_examples)
      features = [example for (example, _) in labeled_examples]
      labels = [label for (_, label) in labeled_examples]
      num_batches = len(features) // FLAGS.batch_size
      features = np.array_split(features, num_batches)
      labels = np.array_split(labels, num_batches)

      this_epoch_loss = 0
      for fs, ls in tqdm(zip(features, labels)):
        [num_steps, loss_value, _] = sess.run(
          [global_step, loss, train_op],
          feed_dict={features_input: fs, labels_input: ls})
        this_epoch_loss += loss_value
        total_loss += loss_value
        if num_steps % FLAGS.log_step == 0:
          print('Step %d: loss %g\ttotal loss %g' % (num_steps, loss_value, total_loss / num_steps))
          saver.save(sess, "vggish_finetuned/model.ckpt", global_step=num_steps)
      
      print('Epoch %d Loss:', this_epoch_loss / len(features))
    
    saver.save(sess, "vggish_finetuned/model.ckpt")

if __name__ == '__main__':
  tf.app.run()
