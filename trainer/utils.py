"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import scipy.misc
import numpy as np
from PIL import Image
import tensorflow as tf
from google.cloud import storage
import tempfile

temp = tempfile.NamedTemporaryFile()
client = storage.Client()
bucket = None

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def get_bucket(bucket_name):
    global bucket
    bucket = client.get_bucket(bucket_name)
    return bucket

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              is_crop=False, is_grayscale=False):
    return tf.Session().run(tf.image.random_flip_left_right(transform(imread(image_path, is_grayscale), input_height, input_width, resize_height, resize_width, is_crop)))

def save_images(images, size, image_path, is_grayscale=False):
    return imsave(inverse_transform(images), size, image_path, is_grayscale)

def imread(path, is_grayscale=False):
    if (is_grayscale):
        temp.seek(0,0)
        path.download_to_file(temp)
        return scipy.misc.imread(temp, flatten = True).astype('uint8')
    else:
        temp.seek(0,0)
        path.download_to_file(temp)
        return scipy.misc.imread(temp).astype('uint8')

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def merge_gray(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w] = image
    return img

def imsave(images, size, path, is_grayscale=False):
    if (is_grayscale):
        blob = bucket.blob(path)
        temp.seek(0,0)
        scipy.misc.imsave(temp, merge_gray(images, size), "jpeg")
        temp.seek(0,0)
        return blob.upload_from_file(temp,content_type='image/jpeg')
    else:
        blob = bucket.blob(path)
        temp.seek(0,0)
        scipy.misc.imsave(temp, merge(images, size), "jpeg")
        temp.seek(0,0)
        return blob.upload_from_file(temp,content_type='image/jpeg')

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, is_crop=True):
    if is_crop:
        cropped_image = center_crop(image, input_height, input_width, resize_h=resize_height, resize_w=resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)/127.5 - 1.
  # return cropped_image

def inverse_transform(images):
    return (images+1.)/2.

def make_gif(images, fname, duration=2, true_image=False):
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images)/duration*t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x+1)/2*255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    blob = bucket.blob(fname)  # should be name.gif
    temp.seek(0,0)
    clip.write_gif(temp.name, fps = len(images) / duration)
    temp.seek(0,0)
    blob.upload_from_file(temp, content_type='image/gif')
