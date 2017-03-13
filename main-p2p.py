# Adpatation from yenchenlin's pix2pix-tensorflow WIP
from __future__ import division
import os
import sys
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

# Use a CGAN with layer_size y class labels for the y_dim and concat it to the inputs of the generator and discriminator

class txt2pic():
    def __init__(self, image_size=256, batch_size=64, sample_size=1):

        self.sess = tf.Session()
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.image_size = image_size
        self.output_size = image_size
        self.y_dim = 1539  # Number of Unique tags
        self.z_dim = 100  # Should represent noise

        self.gf_dim = 64
        self.df_dim = 64
        self.gfc_dim = 1024
        self.dfc_dim = 1024
        self.c_dim = 3 # 1 for grayscale

        self.L1_lambda = 100
        # try out Elastic Nets
        # Declare the elastic net loss function
        # elastic_param1 = tf.constant(1.)
        # elastic_param2 = tf.constant(1.)
        # l1_a_loss = tf.reduce_mean(tf.abs(A))
        # l2_a_loss = tf.reduce_mean(tf.square(A))
        # e1_term = tf.multiply(elastic_param1, l1_a_loss)
        # e2_term = tf.multiply(elastic_param2, l2_a_loss)
        # loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.checkpoint_dir = "./checkpoint"
        self.build_model()

    def build_model(self):
        self.tags= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='tags')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        image_dims = [self.image_size, self.image_size, self.c_dim]
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.G = self.generator(self.z, self.tags)

        self.D, self.D_logits = self.discriminator(self.inputs, self.tags, reuse=False) # Real
        self.D_, self.D_logits_ = self.discriminator(self.G, self.tags, reuse=True)     # Fake

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake
         self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_))) + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()

    def train(self):
        tf.initialize_all_variables().run()

    def generator(self, z, tags):
        s_h, s_w = self.output_size, self.output_size
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(tags, [self.batch_size, 1, 1, self.y_dim])
        z = tf.concat([z, tags], 1)

        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = tf.concat([h0, tags], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, tags)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, tags)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def discriminator(self, image, tags, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        tags_reshape = tf.reshape(tags, [self.batch_size, 1, 1, self.y_dim])
        x = tf.concat(3, [image, tags_reshape])

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, tags_reshape)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])
        h1 = tf.concat([h1, tags], 1)

        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = tf.concat([h2, tags], 1)

        h3 = linear(h2, 1, 'd_h3_lin')

        return tf.nn.sigmoid(h3), h3

if __name__ == '__main__':
    try:
        cmd = sys.argv[1]
        if cmd == "train":
            model = txt2pic()
            model.train()
        elif cmd == "test":
            try:
                size = sys.argv[2]
                model = txt2pic(image_size=size)
                c.sample()
            except IndexError:
                size = sys.argv[2]
                model = txt2pic()
                model.sample()
        else:
            print("Usage: python main.py [train, test, (optional) img output size]")
    except IndexError:
        print("Usage: python main.py [train, test, (optional) img output size]")
        sys.exit()
