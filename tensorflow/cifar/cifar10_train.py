# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
from tensorflow.python.client import timeline
from keras import datasets as dset
import numpy as np

import cifar10

import os
#os.environ["CUDA_VISIBLE_DEVICES"]='0'


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 390,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 30,
                            """How often to log results to the console.""")

EPOCH_SIZE = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_LABELS = cifar10.NUM_CLASSES

def normalize(X_train,X_test):
    #this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def predt(sess, x_test, y_test, logits, x, y):
    size = x_test.shape[0]
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    for begin in xrange(0, size, FLAGS.batch_size):
        end = begin + FLAGS.batch_size
        if end <= size:
            predictions[begin:end, :] = sess.run(
                logits,
                feed_dict={x: x_test[begin:end, ...], y: y_test[begin:end]})
        else:
            batch_predictions = sess.run(
                logits,
                feed_dict={x: x_test[-FLAGS.batch_size:, ...], y: y_test[-FLAGS.batch_size:]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]

    correct = 0
    pred = []
    for item in predictions:
      pred.append(np.argmax(softmax(item)))
    for i in range(len(pred)):
#            print ("i=", i)
#            print ("pred and y_test:", pred[i], y_test[i][0])
        if pred[i] == y_test[i]:
            correct += 1
    acc = (1.0000 * correct / predictions.shape[0])
    print ("acc:", acc)

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
#      images, labels = cifar10.distorted_inputs()
      images, labels = cifar10.inputs(False)
      
    (x_train, y_train_orl), (x_test, y_test_orl) = dset.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)

    y_train_orl = y_train_orl.astype('int32')
    y_test_orl = y_test_orl.astype('int32')
    y_train_flt = y_train_orl.ravel()
    y_test_flt = y_test_orl.ravel()
    
    print("image and lables:", images, labels)
    print ("xtrian and ytrain:", type(x_train), x_train.shape, x_train.dtype, type(y_train_orl), y_train_orl.shape, y_train_orl.dtype)
#    exit(0)

    x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 32,32,3))
    y = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = cifar10.inference(x)

    # Calculate loss.
    loss = cifar10.loss(logits, y)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0 and self._step > 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      step = 0
      f = open('tl.json', 'w')
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      time_begin = time.time()
      while not mon_sess.should_stop():
        offset = (step * FLAGS.batch_size) % (EPOCH_SIZE - FLAGS.batch_size)
        x_data = x_train[offset:(offset + FLAGS.batch_size), ...]
        y_data_flt = y_train_flt[offset:(offset + FLAGS.batch_size)]
        mon_sess.run(train_op, feed_dict={x:x_data, y:y_data_flt})#, options=run_options, run_metadata=run_metadata)
#        tl = timeline.Timeline(run_metadata.step_stats)
 #       ctf = tl.generate_chrome_trace_format()
  #    f.write(ctf)
        step += 1
        if (step+1 == FLAGS.max_steps):
          predt(mon_sess, x_test, y_test_flt, logits, x, y)
      time_end = time.time()
      training_time = time_end - time_begin
      print("Training elapsed time: %f s" % training_time)
      f.close()


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
