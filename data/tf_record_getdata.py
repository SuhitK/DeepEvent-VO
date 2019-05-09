from __future__ import print_function
import os
import numpy as np
import tensorflow as tf

_MAX_SKIP_FRAMES = 1
_TEST_SKIP_FRAMES = 1
n_frames = 1

# %matplotlib inline

import tensorflow as tf

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

base_path = "./3/"

# tfrecords_filename_left = './mvsec_data/outdoor_day1_diff/left_event_images.tfrecord'
# tfrecords_filename_right = './mvsec_data/outdoor_day1_diff/right_event_images.tfrecord'

tfrecords_filename_left = base_path + 'left_event_images.tfrecord'
tfrecords_filename_right = base_path + 'right_event_images.tfrecord'

def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = {
            'image_iter': tf.FixedLenFeature([], tf.int64),
            'shape': tf.FixedLenFeature([], tf.string),
            'event_count_images': tf.FixedLenFeature([], tf.string),
            'event_time_images': tf.FixedLenFeature([], tf.string),
            'image_times': tf.FixedLenFeature([], tf.string),
            'prefix': tf.FixedLenFeature([], tf.string),
            'cam': tf.FixedLenFeature([], tf.string)
        }

    data = tf.parse_single_example(
      serialized_example, features
      # Defaults are not specified since both keys are required.
      )

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].

    shape = tf.decode_raw(data['shape'], tf.uint16)
    shape = tf.cast(shape, tf.int32)

    event_count_images = tf.decode_raw(data['event_count_images'], tf.uint16)
    event_count_images = tf.reshape(event_count_images, shape)
    event_count_images = tf.cast(event_count_images, tf.float32)
    # print (event_count_image)
    event_count_image = event_count_images[:n_frames, :, :, :]
    event_count_image = tf.reduce_sum(event_count_image, axis=0)

    event_time_images = tf.decode_raw(data['event_time_images'], tf.float32)        
    event_time_images = tf.reshape(event_time_images, shape)
    event_time_images = tf.cast(event_time_images, tf.float32)
    # print (event_time_image)
    event_time_image = event_time_images[:n_frames, :, :, :]
    event_time_image = tf.reduce_max(event_time_image, axis=0)
    
    # Normalize timestamp image to be between 0 and 1.
    event_time_image /= tf.reduce_max(event_time_image)

    event_image = tf.concat([event_count_image, event_time_image], 2)
    event_image = tf.cast(event_image, tf.float32)
    event_image = tf.image.resize_image_with_crop_or_pad(event_image, 
                                                         IMAGE_HEIGHT, 
                                                         IMAGE_WIDTH)
    event_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 4])

    return event_image

filename_queue_left = tf.train.string_input_producer([tfrecords_filename_left], name="queue")
event_image_left = read_and_decode(filename_queue_left)

filename_queue_right = tf.train.string_input_producer([tfrecords_filename_right], name="queue")
event_image_right = read_and_decode(filename_queue_right)

with tf.Session()  as sess:

    for i in xrange(4444):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        img_left = sess.run(event_image_left)    
        # np.save('./mvsec_data/outdoor_day1_diff/' + 'left_event' + str(i).zfill(5) + '.npy', img_left)
        np.save(base_path + 'left_event' + str(i).zfill(5) + '.npy', img_left)

        img_right = sess.run(event_image_right)    
        # np.save('./mvsec_data/outdoor_day1_diff/' + 'right_event' + str(i).zfill(5) + '.npy', img_right) 
        np.save(base_path + 'right_event' + str(i).zfill(5) + '.npy', img_right) 

    coord.request_stop()
    coord.join(threads)

