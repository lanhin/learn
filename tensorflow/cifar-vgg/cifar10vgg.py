
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers

import cf10
import tensorflow as tf
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
flags.DEFINE_integer("train_steps", 1000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")

FLAGS = flags.FLAGS

alpha = 0.1
tau = 1

class cifar10vgg:
    def __init__(self,train=True):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            pass
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 4
        learning_rate = 0.1
        lr_decay = 1e-6

        BATCH_SIZE = 128
        EPOCH_SIZE = 50000
        NUM_EPOCHS = 250
        NUM_LABELS = 10

        # The data, shuffled and split between train and test sets:
        (x_train, y_train_orl), (x_test, y_test_orl) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.to_categorical(y_train_orl, self.num_classes)
        y_test = keras.utils.to_categorical(y_test_orl, self.num_classes)
        y_train_flt = y_train_orl.ravel()

        lrf = learning_rate

        '''
        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)
        '''
        x = tf.placeholder(tf.float32, shape=(None, 32,32,3))
        y = tf.placeholder(tf.float32, shape=(None,))
        logits = model(x)
        #softout = Activation('softmax')(logits)
        #cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        #cross_entropy2 = tf.reduce_mean(categorical_crossentropy(y, logits))
        global_step = tf.contrib.framework.get_or_create_global_step()
        loss = cf10.loss(logits, y)
        train_op = cf10.train(loss,global_step)

        #lr = 0.1
        #opt = tf.train.GradientDescentOptimizer(lrf)
        #train_op = opt.minimize(cross_entropy2)

        init_op = tf.global_variables_initializer()

        sess = tf.Session()
        K.set_session(sess)
        sess.run(init_op)
        
        #optimization details
        #sgd = optimizers.SGD(lr=lrf, decay=lr_decay, momentum=0.9, nesterov=True)
        #model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

        def predt(x_test, y_test, logits):
            size = x_test.shape[0]
            predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
            for begin in xrange(0, size, BATCH_SIZE):
                end = begin + BATCH_SIZE
                if end <= size:
                    predictions[begin:end, :] = sess.run(
                        logits,
                        feed_dict={x: x_test[begin:end, ...], K.learning_phase(): 0})
                else:
                    batch_predictions = sess.run(
                        logits,
                        feed_dict={x: x_test[-BATCH_SIZE:, ...], K.learning_phase(): 0})
                    predictions[begin:, :] = batch_predictions[begin - size:, :]

            correct = 0
            pred = np.argmax(predictions, 1)
            for i in range(len(pred)):
    #            print ("i=", i)
    #            print ("pred and y_test:", pred[i], y_test[i][0])
                if pred[i] == y_test[i][0]:
                    correct += 1
            acc = (1.0000 * correct / predictions.shape[0])

            print ("acc:", acc)

        # training process in a for loop with learning rate drop every 25 epoches.
        print ("start training...")
#        K.set_learning_phase(1) # set train phase
        time_begin = time.time()
        print("Training begins @ %f" % time_begin)

        for step in xrange(int(NUM_EPOCHS * EPOCH_SIZE) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (EPOCH_SIZE - BATCH_SIZE)
            x_data = x_train[offset:(offset + BATCH_SIZE), ...]
            y_data = y_train[offset:(offset + BATCH_SIZE)]
            y_data_flt = y_train_flt[offset:(offset + BATCH_SIZE)]
            y_data_orl = y_train_orl[offset:(offset + BATCH_SIZE)]
            #print ("y_data, y_data_orl, y_data_flt:", y_data, y_data_orl, y_data_flt)
            sess.run(train_op, feed_dict={x:x_data, y:y_data_flt,K.learning_phase(): 1 })
            #model.train_on_batch(x_data, y_data)
            if step > 0 and step % 390 == 0:
                print ("loss:", sess.run(loss, feed_dict={x:x_data, y:y_data_flt,K.learning_phase(): 1}), end=' ')
                predt(x_test, y_test_orl, logits)

        time_end = time.time()
        print("Training ends @ %f" % time_end)
        training_time = time_end - time_begin
        print("Training elapsed time: %f s" % training_time)

        model.save_weights('cifar10vgg.h5')

        # start test
        #print (K.learning_phase())
        predt(x_test, y_test_orl, logits)
        #sess.close()

        return model

if __name__ == '__main__':

#    if FLAGS.job_name is None or FLAGS.job_name == "":
#        raise ValueError("Must specify an explicit `job_name`")
#    if FLAGS.task_index is None or FLAGS.task_index =="":
#        raise ValueError("Must specify an explicit `task_index`")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = cifar10vgg()
    '''
    summary = model.model.summary()
    print (summary)
    layer_index = 0
    for layer in model.model.layers:
        layer_weights = layer.weights
        if layer_weights:
            print ("layer index:", layer_index)
            for weights in layer_weights:
#                print (weights)
                print (weights.name, weights.shape, weights.dtype)

        layer_index += 1
    '''
    '''
    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1) == np.argmax(y_test,1)

    print (residuals, sum(residuals), len(residuals))
    loss = 1.0000 * sum(residuals)/len(residuals)
    print("the validation acc is: ",loss)
    '''
