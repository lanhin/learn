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

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 390*250,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 30,
                            """How often to log results to the console.""")
#lanhin
tf.app.flags.DEFINE_integer("task_index", None,
                            "Worker task index, should be >= 0. task_index=0 is "
                            "the master worker task the performs the variable "
                            "initialization ")
tf.app.flags.DEFINE_integer("num_gpus", 0,
                            "Total number of gpus for each machine."
                            "If you don't use GPU, please set it to '0'")
tf.app.flags.DEFINE_string("ps_hosts","localhost:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", None,"job name: worker or ps")
tf.app.flags.DEFINE_integer("tau", 2, "The Tau value")

if FLAGS.job_name == "ps": 
  os.environ["CUDA_VISIBLE_DEVICES"]=''
#elif FLAGS.task_index == 0:
#  os.environ["CUDA_VISIBLE_DEVICES"]='0'
#else:
#  os.environ["CUDA_VISIBLE_DEVICES"]='1'

alpha = 0.1

EPOCH_SIZE = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_LABELS = cifar10.NUM_CLASSES

#lanhin end

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
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()  # may this can fix the exp() overflow problem.  --lanhin
    #return np.exp(x) / np.sum(np.exp(x), axis=0)

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

   #lanhin
  #Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)

  cluster = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})

  server = tf.train.Server(
    cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
  if FLAGS.job_name == "ps":
    server.join()
  # only worker will do train()
  is_chief = False
  if FLAGS.task_index == 0:
    is_chief = True

  #lanhin end

  #with tf.Graph().as_default():
  
  # Use comment to choose which way of tf.device() you want to use
  #with tf.Graph().as_default(), tf.device(tf.train.replica_device_setter(
  #    worker_device="/job:worker/task:%d" % FLAGS.task_index,
  #    cluster=cluster)):
  with tf.device("job:worker/task:%d" % FLAGS.task_index):
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    #with tf.device('/cpu:0'):
    images, labels = cifar10.distorted_inputs()

    (x_train, y_train_orl), (x_test, y_test_orl) = dset.cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = normalize(x_train, x_test)

    y_train_orl = y_train_orl.astype('int32')
    y_test_orl = y_test_orl.astype('int32')
    y_train_flt = y_train_orl.ravel()
    y_test_flt = y_test_orl.ravel()

    x = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, 32,32,3))
    y = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #logits, local_var_list = cifar10.inference(images)
    logits, local_var_list = cifar10.inference(x)

    # Calculate loss.
    #loss = cifar10.loss(logits, labels)
    loss = cifar10.loss(logits, y)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    # the temp var part, for performance testing
    tmp_var_list = []
    var_index = 0
    for var in local_var_list:
      var_index += 1
      tmp_var_list.append(tf.Variable(tf.zeros(var.shape), name="tmp_var"+str(var_index)))

    # the non chief workers get local var init_op here
    if not is_chief:
      init_op = tf.global_variables_initializer()
    else:
      init_op = None

    # start global variables region
    global_var_list = []
    with tf.device("/job:ps/replica:0/task:0/cpu:0"):
      # barrier var
      finished = tf.get_variable("worker_finished",[],tf.int32,tf.zeros_initializer(tf.int32),trainable=False)                    
      with finished.graph.colocate_with(finished):
        finish_op = finished.assign_add(1,use_locking=True)

      var_index = 0
      for var in local_var_list:
        var_index += 1
        global_var_list.append(tf.Variable(tf.zeros(var.shape), name="glo_var"+str(var_index)))

    def assign_global_vars(): # assign local vars' values to global vars
      return [gvar.assign(lvar) for (gvar, lvar) in zip(global_var_list, local_var_list)]
    
    def assign_local_vars(): # assign global vars' values to local vars
      return [lvar.assign(gvar) for (gvar, lvar) in zip(global_var_list, local_var_list)]

    def assign_tmp_vars(): # assign local vars' values to tmp vars
      return [tvar.assign(lvar) for (tvar, lvar) in zip(tmp_var_list, local_var_list)]

    def assign_local_vars_from_tmp(): # assign tmp vars' values to local vars
      return [lvar.assign(tvar) for (tvar, lvar) in zip(tmp_var_list, local_var_list)]

    def update_before_train(alpha, w, global_w):
      varib = alpha*(w-global_w)
      gvar_op = global_w.assign(global_w + varib)
      return gvar_op, varib
      
    def update_after_train(w, vab):
      return w.assign(w-vab)

    assign_list_local = assign_local_vars()
    assign_list_global = assign_global_vars()
    assign_list_loc2tmp = assign_tmp_vars()
    assign_list_tmp2loc = assign_local_vars_from_tmp()

    before_op_tuple_list = []
    after_op_tuple_list = []
    vbholder_list = []
    for (gvar, lvar) in zip(global_var_list, local_var_list):
      before_op_tuple_list.append((update_before_train(alpha, lvar, gvar)))
    for var in local_var_list:
      vbholder_list.append(tf.placeholder("float", var.shape))
      after_op_tuple_list.append((update_after_train(var, vbholder_list[-1]), vbholder_list[-1]))

    # the chief worker get global var init op here
    if is_chief:
      init_op = tf.global_variables_initializer()
    
    # global variables region end

    #lanhin start
    sv = tf.train.Supervisor(
      is_chief=True,#is_chief,
      logdir=FLAGS.train_dir,
      init_op=init_op,
      #local_init_op=loc_init_op,
      recovery_wait_secs=1)
    #global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    if is_chief:
      sess.run(assign_list_global)
      barrier_finished = sess.run(finish_op)
      print ("barrier_finished:", barrier_finished)
    else:
      barrier_finished = sess.run(finish_op)
      print ("barrier_finished:", barrier_finished)
    while barrier_finished < num_workers:
      time.sleep(1)
      barrier_finished = sess.run(finished)
    sess.run(assign_list_local)
    print("Worker %d: Session initialization complete." % FLAGS.task_index)
      # lanhin end
    
    #sess = tf.Session()
    #sess.run(init_op)
    #tf.train.start_queue_runners(sess)
    f = open('tl_dist.json', 'w')
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    time_begin = time.time()
#    while not mon_sess.should_stop():
#      mon_sess.run(train_op)
    for step in range(FLAGS.max_steps):
      offset = (step * FLAGS.batch_size) % (EPOCH_SIZE - FLAGS.batch_size)
      x_data = x_train[offset:(offset + FLAGS.batch_size), ...]
      y_data_flt = y_train_flt[offset:(offset + FLAGS.batch_size)]

      if step % FLAGS.log_frequency == 0:
        time_step = time.time()
        steps_time = time_step - time_begin
        print ("step:", step, " steps time:", steps_time, end='  ')
        sess.run(assign_list_loc2tmp)
        sess.run(assign_list_local)
        predt(sess, x_test, y_test_flt, logits, x, y)
        sess.run(assign_list_tmp2loc)
        time_begin = time.time()
      if step % FLAGS.tau == 0 and step > 0: # update global weights
        thevarib_list = []
        for i in range(0, len(before_op_tuple_list)):
          (gvar_op, varib) = before_op_tuple_list[i]
          _, thevarib = sess.run([gvar_op, varib])
          thevarib_list.append(thevarib)

        sess.run(train_op, feed_dict={x:x_data, y:y_data_flt})

        for i in range(0, len(after_op_tuple_list)):
          (lvar_op, thevaribHolder) = after_op_tuple_list[i]
          sess.run(lvar_op, feed_dict={thevaribHolder: thevarib_list[i]})

      else:
        sess.run(train_op, feed_dict={x:x_data, y:y_data_flt})#, options=run_options, run_metadata=run_metadata)
        #tl = timeline.Timeline(run_metadata.step_stats)
        #ctf = tl.generate_chrome_trace_format()
    #f.write(ctf)
    time_end = time.time()
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)
    f.close()
    sess.run(assign_list_local)
    predt(sess, x_test, y_test_flt, logits, x, y)

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
