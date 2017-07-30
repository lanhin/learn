#
# lanhin@github.com
# @2017-7-13
#
# ====================

import tensorflow as tf
import numpy as np
import time

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

alpha = 0.1 # alpha, move it into options
tau = 16 # updates interval

def model(X, w):
    return tf.multiply(X, w)

def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
#    with tf.device(tf.train.replica_device_setter(
 #       worker_device="/job:worker/task:%d" % FLAGS.task_index,
  #      ps_device="/job:ps/cpu:0",
   #     cluster=cluster)):
     with tf.device("job:worker/task:%d" % FLAGS.task_index):
      global_step = tf.Variable(0, name="global_step", trainable=False)
      # Build model...

      #Prepare data
      trainX = np.linspace(-1, 1, 101)
      trainY = 2 * trainX + np.random.randn(*trainX.shape) * 0.33

      X = tf.placeholder('float')
      Y = tf.placeholder('float')

      w = tf.Variable(0.0, name = 'weights')
      with tf.device("/job:ps/cpu:0"):
          global_w = tf.get_variable('global_weight', [], initializer=tf.constant_initializer(0))
      y_ = model(X, w)
      loss = tf.square(Y - y_)

#      train_op = tf.train.AdagradOptimizer(0.01).minimize(
#          loss, global_step=global_step)

      train_op = tf.train.GradientDescentOptimizer(0.01).minimize(
          loss, global_step=global_step)


      #diff = w - global_w
      varib = alpha*(w-global_w)
      lvar_op = w.assign(w - varib)
      vab = tf.placeholder('float')
      lvar = w.assign(w-vab)
      gvar_op = global_w.assign(global_w + varib)
      
      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

      # Create a "supervisor", which oversees the training process.
      sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

      sess_config = tf.ConfigProto(
          allow_soft_placement=True,
          log_device_placement=True,
          device_filters=["/job:ps", "/job:worker/task:%d" % FLAGS.task_index])

#      sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

      # The supervisor takes care of session initialization, restoring from
      # a checkpoint, and closing when done or an error occurs.
      #     with sv.managed_session(server.target) as sess:
      # Loop until the supervisor shuts down or 1000000 steps have completed.
      with sv.managed_session(server.target, config=sess_config) as sess:
          time_begin = time.time()
          local_step = 0
          while not sv.should_stop() and local_step < 10000:
              # Run a training step asynchronously.
              # See `tf.train.SyncReplicasOptimizer` for additional details on how to
              # perform *synchronous* training.
              for (x, y) in zip(trainX, trainY):
                  if (local_step % tau == 0):
                      #print "[bf] w, global_w: "+", ".join(str(e) for e in sess.run([w, global_w]))
                      _, thevarib = sess.run([gvar_op, varib])
                      #print "thevarib:",thevarib
                      #print "[bf2] w, global_w: "+", ".join(str(e) for e in sess.run([w, global_w]))
                      _, step = sess.run([train_op, global_step], feed_dict={X:x, Y:y})
                      #print "[md] w, global_w: "+", ".join(str(e) for e in sess.run([w, global_w]))
                      sess.run(lvar, feed_dict={vab:thevarib})
                      local_step += 1
                      #print "[af] w, global_w: "+", ".join(str(e) for e in sess.run([w, global_w]))
                  else:
                      _, step = sess.run([train_op, global_step], feed_dict={X:x, Y:y})
                      local_step += 1
              print(" Worker %d: training step %d done (global step: %d)" %
                    (FLAGS.task_index, local_step, step))
              #the_loss = sess.run(loss, feed_dict={X:1.0, Y:2.0})
              the_diff = sess.run(global_w)
              the_diff = the_diff-2.0
              the_diff *= the_diff
              print "the diff:"+str(the_diff)
              if (the_diff < 0.0005):
                  print "exit..."
                  break

          time_end = time.time()
          training_time = time_end - time_begin
          print("Training elapsed time: %f s" % training_time)

        # Ask for all the services to stop.
      sv.stop()

if __name__ == "__main__":
  tf.app.run()
