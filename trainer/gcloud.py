import os
import numpy as np
import tensorflow as tf
from PIL import Image


# data = glob(os.path.join("gs://dcgan-161707-mlengine/birds", "*.jpg"))[:542]  #CUB BIRD DATASET -- download and put first 10 class birds into "/birds"
# print(data)

# filename_queue = tf.train.string_input_producer(['gs://dcgan-161707-mlengine/birds/Black_Footed_Albatross_0001_796111.jpg'])
# filename_queue = tf.train.string_input_producer(['gs://dcgan-161707-mlengine/data/adult.data.csv'])
filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("./birds/*.jpg"))

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.
#
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)

  # for i in range(1): #length of your filename list

  image = sess.run([my_img]) #here is your image Tensor :)
  # print(image[0].shape)
  Image.fromarray(np.asarray(image[0])).show()

  coord.request_stop()
  coord.join(threads)

# client = storage.Client()

# bucket = client.get_bucket('dcgan-161707-mlengine')

# blob = bucket.get_blob('birds/Black_Footed_Albatross_0001_796111.jpg')
# print(blob)
# blob.download_to_filename('tests.png')
# iterator = bucket.list_blobs(prefix='birds')
# print np.array(iterator)
# print(list(iterator))
# blobs = []
# for i, my_item in enumerate(iterator):
    # my_item.download_to_filename(my_item.name)
    # blobs.append(my_item)
# list(iterator)[1].download_to_file(b)
# print(len(list(iterator)))
# blobs[1].download_to_filename(b)
