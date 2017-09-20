#-*- coding: utf-8 -*-
#File: cifar10_multi_gpu.py
#Author: yobobobo(zhouboacmer@qq.com)
import tensorgo.benchmark.cifar10.cifar10 as cifar10
import tensorflow as tf


def test_inptus():
  cifar10.maybe_download_and_extract()
  with tf.device('/cpu:0'):
    images, labels = cifar10.distorted_inputs()
    batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue(
        [images, labels], capacity=10,
        )
    image_batch, label_batch = batch_queue.dequeue()
  sess = tf.Session()
  images, labels = sess.run([image_batch, label_batch])
  print images

if __name__ == '__main__':
  test_inptus()
