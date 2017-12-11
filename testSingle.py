
from __future__ import print_function

import numpy as np
#from scipy.io import wavfile
#import six
import tensorflow as tf
import os, sys

import utils
import vggish_params
import vggish_slim
import vggish_input


examples_batch = vggish_input.wavfile_to_examples('SingleTestSounds_16bit/90.wav')

  # restricting memory usage, TensorFlow is greedy and will use all memory otherwise
config = tf.ConfigProto()
  #config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
  #config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.90

with tf.Graph().as_default(), tf.Session(config=config) as sess:

    vggish_slim.define_vggish_slim(training=False) # Defines the VGGish TensorFlow model.
    vggish_slim.load_vggish_slim_checkpoint(sess, 'vggish_model.ckpt') # Loads a pre-trained VGGish-compatible checkpoint.

    # locate input and output tensors.
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    feed_dict = {features_tensor: examples_batch}

    [embedding_batch] = sess.run([embedding_tensor], feed_dict=feed_dict)

print('example_batch shape: ', examples_batch.shape)


print(embedding_batch.shape)

for i in range(1,len(embedding_batch)):
    embedding_list = [embedding_batch[i]]

    gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
     # load the trained network from a local drive
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph("C:/tmp/audio_classifier.meta")
        saver.restore(sess,tf.train.latest_checkpoint('C:/tmp/'))

        # Now, let's access and create placeholders variables and
        # create feed-dict to feed new data

        graph = tf.get_default_graph()
        x_pl = graph.get_tensor_by_name("xPlaceholder:0")
        feed_dict = {x_pl: embedding_list}

        #Now, access the op that you want to run.
        op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

        y_pred = sess.run(op_to_restore, feed_dict)[0]

        #print(y_pred)
        pred = sess.run(tf.argmax(y_pred, axis=0))
        print("class predicion embedding 1:", pred)
