from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.append(os.path.join('.', '..'))
import utils
import tensorflow as tf
import numpy as np

tfrecords_filename = 'singleTest.tfrecords'
record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
embedding_tot =  np.zeros((1, 128))

for string_record in record_iterator:

    example = tf.train.Example()
    example.ParseFromString(string_record)

    embedding_string = (example.features.feature['val/embedding']
                                  .bytes_list
                                  .value[0])

    embedding_1d = np.fromstring(embedding_string, dtype=np.float32)
    reconstructed_embedding = embedding_1d.reshape((-1, 128))
    embedding_tot = np.append(embedding_tot,reconstructed_embedding, axis=0)

embedding_tot = np.delete(embedding_tot, 0, axis=0)
print(embedding_tot.shape)
print("tfRecord uploaded!")
print(len(embedding_tot))
for i in range(1,len(embedding_tot)):
    embedding_list = [embedding_tot[i]]

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
