# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Distributed MNIST training and validation, with model replicas.

A simple softmax model with one hidden layer is defined. The parameters
(weights and biases) are located on two parameter servers (ps), while the
ops are defined on a worker node. The TF sessions also run on the worker
node.
Multiple invocations of this script can be done in parallel, with different
values for --task_index. There should be exactly one invocation with
--task_index, which will create a master session that carries out variable
initialization. The other, non-master, sessions will wait for the master
session to finish the initialization before proceeding to the training stage.

The coordination between the multiple worker invocations occurs due to
the definition of the parameters on the same ps devices. The parameter updates
from one worker is visible to all other workers. As such, the workers can
perform forward computation and gradient calculation in parallel, which
should lead to increased training speed for the simple model.
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys
import os
import tempfile
import time
import numpy

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


flags = tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_boolean("download_only", False,
                     "Only perform downloading of data; Do not proceed to "
                     "session preparation, model definition or training")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("num_gpus", 0,
                     "Total number of gpus for each machine."
                     "If you don't use GPU, please set it to '0'")
flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")
flags.DEFINE_integer("train_steps", 10000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 128, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_integer("tau", 2, "The Tau value")

FLAGS = flags.FLAGS

if FLAGS.job_name == "ps": 
  os.environ["CUDA_VISIBLE_DEVICES"]=''
elif FLAGS.task_index == 0:
  os.environ["CUDA_VISIBLE_DEVICES"]='0'
else:
  os.environ["CUDA_VISIBLE_DEVICES"]='1'

IMAGE_PIXELS = 28

alpha = 0.1
#tau = 1

def main(unused_argv):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  if FLAGS.download_only:
    sys.exit(0)

  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

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

  is_chief = (FLAGS.task_index == 0)
  if FLAGS.num_gpus > 0:
    if FLAGS.num_gpus < num_workers:
      raise ValueError("number of gpus is less than number of workers")
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu 
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)
  # The device setter will automatically place Variables ops on separate
  # parameter servers (ps). The non-Variable ops will be placed on the workers.
  # The ps use CPU and workers use corresponding GPU
#  with tf.device(
#      tf.train.replica_device_setter(
#          worker_device=worker_device,
#          ps_device="/job:ps/cpu:0",
#          cluster=cluster)):
  with tf.device("job:worker/task:%d" % FLAGS.task_index):
#    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Variables of the hidden layer
    hid_w = tf.Variable(
        tf.truncated_normal(
            [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
            stddev=1.0 / IMAGE_PIXELS),
        name="hid_w")
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal(
            [FLAGS.hidden_units, 10],
            stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # Ops: located on the worker specified with FLAGS.task_index
    x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
    y_ = tf.placeholder(tf.float32, [None, 10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

#    opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
    opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

#  with tf.device(
#      tf.train.replica_device_setter(
#          worker_device=worker_device,
#          ps_device="/job:ps/cpu:0",
#          cluster=cluster)):

#    train_step = opt.minimize(cross_entropy, global_step=global_step)
    train_step = opt.minimize(cross_entropy)

    loc_init_op = tf.global_variables_initializer()
    train_dir = tempfile.mkdtemp()
    #Graph end here

    with tf.device("/job:ps/replica:0/task:0/cpu:0"):
          # Variables of the hidden layer
      glo_hid_w = tf.Variable(
        tf.truncated_normal(
          [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
          stddev=1.0 / IMAGE_PIXELS),
        name="glo_hid_w")
      glo_hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="glo_hid_b")

      # Variables of the softmax layer
      glo_sm_w = tf.Variable(
        tf.truncated_normal(
          [FLAGS.hidden_units, 10],
          stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="glo_sm_w")
      glo_sm_b = tf.Variable(tf.zeros([10]), name="glo_sm_b")

    def assign_global_vars():
      return [glo_hid_w.assign(hid_w), glo_hid_b.assign(hid_b), glo_sm_w.assign(sm_w), glo_sm_b.assign(sm_b)]

    def assign_local_vars():
      return [hid_w.assign(glo_hid_w), hid_b.assign(glo_hid_b), sm_w.assign(glo_sm_w), sm_b.assign(glo_sm_b)]

    
    def update_before_train(alpha, w, global_w):
      varib = alpha*(w-global_w)
      gvar_op = global_w.assign(global_w + varib)
      return gvar_op, varib
      
    def update_after_train(w, vab):
      return w.assign(w-vab)

    assign_list_local = assign_local_vars()
    assign_list_global = assign_global_vars()
    
    before_op_tuple_list = []
    after_op_tuple_list = []
    before_op_tuple_list.append((update_before_train(alpha, hid_w, glo_hid_w)))
    before_op_tuple_list.append((update_before_train(alpha, hid_b, glo_hid_b)))
    before_op_tuple_list.append((update_before_train(alpha, sm_w, glo_sm_w)))
    before_op_tuple_list.append((update_before_train(alpha, sm_b, glo_sm_b)))
    vbhw = tf.placeholder("float", [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units])
    after_op_tuple_list.append((update_after_train(hid_w, vbhw), vbhw))
    vbhb = tf.placeholder("float", [FLAGS.hidden_units])
    after_op_tuple_list.append((update_after_train(hid_b, vbhb), vbhb))
    vbsw = tf.placeholder("float", [FLAGS.hidden_units, 10])
    after_op_tuple_list.append((update_after_train(sm_w, vbsw), vbsw))
    vbsb = tf.placeholder("float", [10])
    after_op_tuple_list.append((update_after_train(sm_b, vbsb), vbsb))
    
    init_op = tf.global_variables_initializer()

    sv = tf.train.Supervisor(
      is_chief=is_chief,
      logdir=train_dir,
      init_op=init_op,
      local_init_op=loc_init_op,
      recovery_wait_secs=1)
    #global_step=global_step)

    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True,
        device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

    # The chief worker (task_index==0) session will prepare the session,
    # while the remaining workers will wait for the preparation to complete.
    if is_chief:
      print("Worker %d: Initializing session..." % FLAGS.task_index)
    else:
      print("Worker %d: Waiting for session to be initialized..." %
            FLAGS.task_index)

    sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

    print("Worker %d: Session initialization complete." % FLAGS.task_index)

    # Perform training
    time_begin = time.time()
    print("Training begins @ %f" % time_begin)

    local_step = 0
    tau = FLAGS.tau
    while True:
      # Training feed
      batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
      train_feed = {x: batch_xs, y_: batch_ys}
      if local_step % tau == 0:
        #print ("Update weights...")
        thevarib_list = []
        for i in range(0, len(before_op_tuple_list)):
          (gvar_op, varib) = before_op_tuple_list[i]
          _, thevarib = sess.run([gvar_op, varib])
          thevarib_list.append(thevarib)

        sess.run(train_step, feed_dict=train_feed)

        for i in range(0, len(after_op_tuple_list)):
          (lvar_op, thevaribHolder) = after_op_tuple_list[i]
          sess.run(lvar_op, feed_dict={thevaribHolder: thevarib_list[i]})

      else:
          sess.run(train_step, feed_dict=train_feed)
      local_step += 1

      #now = time.time()
      #print("%f: Worker %d: training step %d done" %
      #      (now, FLAGS.task_index, local_step))

      if local_step >= FLAGS.train_steps:
        break

    time_end = time.time()
    print("Training ends @ %f" % time_end)
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)

    # Validation feed
    def error_rate(predictions, labels):
      """Return the error rate based on dense predictions and sparse labels."""
      numpy.set_printoptions(threshold=10)
      print ('prediction:',numpy.argmax(predictions, 1))
      print ('labels:', numpy.argmax(labels, 1))
      return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

    val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
    val_xent = sess.run(cross_entropy, feed_dict=val_feed)
    sess.run(assign_list_local)
    prediction = sess.run(y, feed_dict=val_feed)
    print('Minibatch error: %.1f%%' % error_rate(prediction, mnist.validation.labels))
    print("After %d training step(s), validation cross entropy = %g" %
          (FLAGS.train_steps, val_xent))


if __name__ == "__main__":
  tf.app.run()
