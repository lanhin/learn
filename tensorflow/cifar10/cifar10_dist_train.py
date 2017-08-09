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

import cifar10

import os

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 60,
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

if FLAGS.job_name == "ps": 
  os.environ["CUDA_VISIBLE_DEVICES"]=''
elif FLAGS.task_index == 0:
  os.environ["CUDA_VISIBLE_DEVICES"]='0'
else:
  os.environ["CUDA_VISIBLE_DEVICES"]='1'

alpha = 0.1
#lanhin end

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
  #lanhin end

  #with tf.Graph().as_default():
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

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, local_var_list = cifar10.inference(images)

    # Calculate loss.
    loss = cifar10.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    loc_init_op = tf.global_variables_initializer()

    # start global variables region
    global_var_list = []
    with tf.device("/job:ps/replica:0/task:0/cpu:0"):
      var_index = 0
      for var in local_var_list:
        var_index += 1
        global_var_list.append(tf.Variable(tf.zeros(var.shape), name="glo_var"+str(var_index)))

    def assign_global_vars():
      return [gvar.assign(lvar) for (gvar, lvar) in zip(global_var_list, local_var_list)]
    
    def assign_local_vars():
      return [lvar.assign(gvar) for (gvar, lvar) in zip(global_var_list, local_var_list)]

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
    vbholder_list = []
    for (gvar, lvar) in zip(global_var_list, local_var_list):
      before_op_tuple_list.append((update_before_train(alpha, lvar, gvar)))
    for var in local_var_list:
      vbholder_list.append(tf.placeholder("float", var.shape))
      after_op_tuple_list.append((update_after_train(var, vbholder_list[-1]), vbholder_list[-1]))

    init_op = tf.global_variables_initializer()
    
    # global variables region end

    '''
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        print ("is this a init?")
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
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
    '''

    #with tf.train.MonitoredTrainingSession(
        # added by lanhin
    #    master=server.target,
    #    is_chief=(FLAGS.task_index==0),
        # added by lanhin end
    #    checkpoint_dir=FLAGS.train_dir,
    #    hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               #tf.train.NanTensorHook(loss),
    #           _LoggerHook()],
    #    config=tf.ConfigProto(
    #        log_device_placement=FLAGS.log_device_placement)) as mon_sess:

    #lanhin start
    is_chief=(FLAGS.task_index==0),
    sv = tf.train.Supervisor(
      is_chief=is_chief,
      logdir=FLAGS.train_dir,
      init_op=init_op,
      local_init_op=loc_init_op,
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

    print("Worker %d: Session initialization complete." % FLAGS.task_index)
    # lanhin end
    
    #sess = tf.Session()
    #sess.run(init_op)
    #tf.train.start_queue_runners(sess)
    time_begin = time.time()
#    while not mon_sess.should_stop():
#      mon_sess.run(train_op)
    for step in range(FLAGS.max_steps):
      if step % FLAGS.log_frequency == 0:
        print ("step:", step)
      sess.run(train_op)
    time_end = time.time()
    training_time = time_end - time_begin
    print("Training elapsed time: %f s" % training_time)



def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
