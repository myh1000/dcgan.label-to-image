# Adpatation from carpedm20's DCGAN-tensorflow
from __future__ import division
import os
import sys
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange  # Compatability w/ Python3

from ops import *
from utils import *

# Use a CGAN with layer_size y class labels for the y_dim and concat it to the inputs of the generator and discriminator

class txt2pic():
    def __init__(self, image_size=256, batch_size=64):

        self.sess = tf.Session()

        self.batch_size = batch_size
        print("batch_size: %d" % self.batch_size)
        self.image_size = image_size
        self.output_size = image_size

        self.y_dim = 10  # Number of Unique tags
        self.z_dim = 100  # Should represent noise

        self.gf_dim = 64
        self.df_dim = 64

        self.gfc_dim = 1024
        self.dfc_dim = 1024

        self.c_dim = 3 # 1 for grayscale

        # try out Elastic Nets
        # Declare the elastic net loss function
        # elastic_param1 = tf.constant(1.)
        # elastic_param2 = tf.constant(1.)
        # l1_a_loss = tf.reduce_mean(tf.abs(A))
        # l2_a_loss = tf.reduce_mean(tf.square(A))
        # e1_term = tf.multiply(elastic_param1, l1_a_loss)
        # e2_term = tf.multiply(elastic_param2, l2_a_loss)
        # loss = tf.expand_dims(tf.add(tf.add(tf.reduce_mean(tf.square(y_target - model_output)), e1_term), e2_term), 0)

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.checkpoint_dir = "./checkpoint"
        self.build_model()

    def build_model(self):
        self.tags = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='tags')
        image_dims = [self.image_size, self.image_size, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(self.z, self.tags)
        self.D, self.D_logits = self.discriminator(self.inputs, self.tags, reuse=False) # Real
        self.D_, self.D_logits_ = self.discriminator(self.G, self.tags, reuse=True)     # Fake

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)

        self.saver = tf.train.Saver()

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        data = glob(os.path.join("dataset", "*.png"))
        counter = 1

        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        batch_idxs = len(data) // self.batch_size
        for epoch in xrange(10000):
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [
                    get_image(batch_file,
                        input_height=self.image_size,
                        input_width=self.image_size,
                        resize_height=self.output_size,
                        resize_width=self.output_size) for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)
                batch_tags = np.zeros((self.batch_size , self.y_dim), dtype=np.float32)
                batch_tags[:,1] = 1  # need actual tags later
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32) # noise

                _, errD_fake, errD_real = self.sess.run([self.d_optim, self.d_loss_fake, self.d_loss_real], feed_dict={self.inputs: batch_images, self.z: batch_z, self.tags: batch_tags})
                _, errG = self.sess.run([self.g_optim, self.g_loss], feed_dict={self.z: batch_z, self.tags: batch_tags})
                # errD_fake = self.d_loss_fake.eval({self.real_data: batch_images})
                # errD_real = self.d_loss_real.eval({self.real_data: batch_images})
                # errG = self.g_loss.eval({self.real_data: batch_images})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d], d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs, errD_fake+errD_real, errG))

                if np.mod(counter, 10) == 1:
                    samples = self.G.eval({self.z: batch_z, self.inputs: batch_images, self.y: batch_tags})
                    save_images(samples, [int(math.sqrt(self.batch_size)), int(math.sqrt(self.batch_size))], './samples/train_{:02d}_{:04d}.png'.format(epoch, idx))
                    print("saved sample")
                if np.mod(counter, 500) == 2:
                  self.save(self.checkpoint_dir, counter)

    def generator(self, z, tags):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_size, self.output_size
            s_h2, s_h4 = int(s_h/2), int(s_h/4)
            s_w2, s_w4 = int(s_w/2), int(s_w/4)

            yb = tf.reshape(tags, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(values=[z, tags], axis=1)

            h0 = tf.nn.relu(
                self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(values=[h0, tags], axis=1)

            h1 = tf.nn.relu(self.g_bn1(
                linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def discriminator(self, image, tags, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            yb = tf.reshape(tags, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = tf.concat(values=[h1, tags], axis=1)

            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
            h2 = tf.concat(values=[h2, tags], axis=1)

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def save(self, checkpoint_dir, step):
        model_name = "txt2pic.model"
        model_dir = "%s_%s" % (self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        model_dir = "%s_%s" % (self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        cmd = sys.argv[1]
        if cmd == "train":
            try:
                size = sys.argv[2]
                model = txt2pic(batch_size=int(size))
                model.train()
            except IndexError:
                size = sys.argv[2]
                model = txt2pic()
                model.train()
        elif cmd == "test":
            try:
                size = sys.argv[2]
                model = txt2pic(image_size=int(size))
                model.test()
            except IndexError:
                size = sys.argv[2]
                model = txt2pic()
                model.test()
        else:
            print("Usage: python main.py [train, test, (optional) img output size]")
    else:
        print("Usage: python main.py [train + (optional) batch_size, test + (optional) img output size]")
