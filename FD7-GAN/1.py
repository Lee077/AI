import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data


class MnistGAN:

    def __init__(self):
        # 定义mnist数据集
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        tf.logging.set_verbosity(old_v)

        # 定义批次大小
        self.batch_size = 64
        # 获取图像大小
        self.img_size = self.mnist.train.images[0].shape[0]
        # 定义分块大小
        self.chunk_size = self.mnist.train.num_examples // self.batch_size
        # 定义迭代次数
        self.epoch_size = 200
        # 定义学习率
        self.lr = 1e-4
        # 真实图片
        self.real_img = tf.placeholder(tf.float32, [None, self.img_size])
        # 生成图片
        self.fake_img = tf.placeholder(tf.float32, [None, self.img_size])
        # leaky Relu 参数
        self.leaky = 0.01
        # 隐藏神经元数量
        self.hideceil = 256
        # 测试样本数量
        self.test = 20

    # 定义生成器网络
    @staticmethod
    def generator(hideceil, img_size, leaky, input):
        with tf.variable_scope("generator"):
            # layerH_arg = tf.get_variable('gh1',tf.truncated_normal([self.img_size, self.hideceil], stddev=0.02))
            layerH = tf.layers.dense(input, hideceil)
            layerR = tf.maximum(layerH, leaky * layerH)
            drop = tf.layers.dropout(layerR, rate=0.5)
            # layer_arg = tf.get_variable('gh2',tf.truncated_normal([self.hideceil, self.img_size], stddev=0.02))
            logits = tf.layers.dense(drop, img_size)
            output = tf.tanh(logits)
            return logits, output

    # 定义鉴别器网络
    @staticmethod
    def discriminator(leaky, hideceil, input, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            # layerH_arg = tf.get_variable('dh1',tf.truncated_normal([self.img_size, self.hideceil], stddev=0.02))
            layer = tf.layers.dense(input, hideceil)
            relu = tf.maximum(leaky * layer, layer)
            # arg = tf.get_variable('dh2',tf.truncated_normal([self.hideceil, 1], stddev=0.02))
            logits = tf.layers.dense(relu, 1)
            output = tf.sigmoid(logits)
            return logits, output

    # 定义损失
    @staticmethod
    def Loss(fake_logits, real_logits):
        # 生成器损失
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))
        # 鉴别器希望对生成图片输出0
        d1_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
        # 鉴别器希望对真实图片输出1
        d2_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
        # 鉴别器总损失
        d_loss = tf.add(d1_loss, d2_loss)
        return g_loss, d_loss

    # 定义优化器
    @staticmethod
    def optimizer(lr, g_loss, d_loss):
        # 获取训练变量
        train_var = tf.trainable_variables()
        # 获取生成器变量
        g_var = [var for var in train_var if 'generator' in var.name]
        # 获取鉴别器变量
        d_var = [var for var in train_var if 'discriminator' in var.name]
        # 定义生成器优化器
        g_optimizer = tf.train.AdamOptimizer(lr).minimize(g_loss, var_list=g_var)
        # 定义鉴别器优化器
        d_optimizer = tf.train.AdamOptimizer(lr).minimize(d_loss, var_list=d_var)
        return g_optimizer, d_optimizer

    # 训练函数
    def train(self):
        # 生成器生成图片
        gen_logits, gen_outpus = self.generator(self.hideceil, self.img_size, self.leaky, self.fake_img)
        # 鉴别器对生成图片鉴别
        fake_logits, fake_outpus = self.discriminator(self.leaky, self.hideceil, gen_outpus)
        # 鉴别器对真实图片鉴别
        real_logits, real_output = self.discriminator(self.leaky, self.hideceil, self.real_img, reuse=True)
        # 获得损失
        g_loss, d_loss = self.Loss(fake_logits, real_logits)
        # 定义优化器
        g_opti, d_opti = self.optimizer(self.lr, g_loss, d_loss)
        # 开始训练

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(0, self.epoch_size):
                for _ in range(0, self.chunk_size):
                    imgs, _ = self.mnist.train.next_batch(self.batch_size)
                    noise_img = np.random.uniform(-1, 1, size=(self.batch_size, self.img_size))

                    sess.run(d_opti, feed_dict={self.real_img: imgs, self.fake_img: noise_img})
                    sess.run(g_opti, feed_dict={self.fake_img: noise_img})

                gen_loss = sess.run(g_loss, feed_dict={self.fake_img: noise_img})
                dis_loss = sess.run(d_loss, feed_dict={self.real_img: imgs, self.fake_img: noise_img})
                print("迭代：" + str(epoch) + " g_loss=" + str(gen_loss) + " d_loss=" + str(dis_loss))
                if (epoch % 5 == 0):
                    noise_img = np.random.uniform(-1, 1, size=(self.test, self.img_size))
                    samples = sess.run(gen_outpus, feed_dict={self.fake_img: noise_img})
                    fig, axes = plt.subplots(figsize=(7, 7), nrows=5, ncols=5, sharey=True, sharex=True)
                    for ax, img in zip(axes.flatten(), samples*(-1)):
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
                    plt.show()

if __name__=='__main__':
    mnist = MnistGAN()
    mnist.train()

