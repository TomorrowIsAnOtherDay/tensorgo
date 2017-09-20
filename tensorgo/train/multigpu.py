#-*- coding: utf-8 -*-
#File: multigpu.py
#Author: yobobobo(zhouboacmer@qq.com)


import numpy as np
import tensorflow as tf
from tensorgo.utils import logger
from tensorgo.train.config import TrainConfig
from tensorgo.tfutils.gradient import record_variable, update_variable, compute_worker_gradient, apply_worker_gradients
import time
import threading

__all__ = ['MultiGpuTrainer']


class MultiGpuTrainer(object):
  def __init__(self, config):
    assert isinstance(config, TrainConfig), type(config)
    self._model = config.model
    self._dataset = config.dataset
    self._config = config

    self.batch_count = 0

    self._work_train_op = []
    self._server_param = []
    self._deltas = []
    self._wrap_delta = []
    self._sync_worker = []

    # specific
    self._probs = []
    self._labels = []
    # end

    self._setup_inputs()
    self._setup()
    self._adam_varlist = []

  def _setup_inputs(self):
    with tf.device('/cpu:0'):
      dataset = self._dataset
      self._data_iter = tf.contrib.data.Iterator.from_structure(dataset.output_types, \
                                                                dataset.output_shapes)
      self._init_data_op = self._data_iter.make_initializer(dataset)

  def update_inputs(self, dataset):
    self._init_data_op = self._data_iter.make_initializer(dataset)
    self._sess.run(self._init_data_op)

  def _setup(self):
    opt = self._model.get_optimizer()
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(self._config.n_towers):
        logger.info('buildding tower:{}'.format(i))
        with tf.device('/gpu:{}'.format(i)):
          with tf.variable_scope('tower_{}'.format(i)):
            loss, prob, label = self._model.build_graph(self._data_iter)
            self._probs.append(prob)
            self._labels.append(label)
            #tf.get_variable_scope().reuse_variables()
            varlist = tf.trainable_variables()
            val_keyword = 'tower_{}'.format(i)
            varlist = filter(lambda x: val_keyword in x.name, varlist)
            # recorde server baseline model
            if i == 0:
              self._server_param = record_variable(varlist)
            #push & pull
            sync_worker = update_variable(self._server_param, varlist)
            delta = compute_worker_gradient(self._server_param, varlist)
            wrap_delta = apply_worker_gradients(opt, self._server_param, delta, scale=(1.0 / self._config.n_towers) * 0.5)
            self._sync_worker.extend(sync_worker)
            self._deltas.extend(delta)
            self._wrap_delta.append(wrap_delta)
            # end
            worker_train_op = opt.minimize(loss)
            self._work_train_op.append(worker_train_op)
    self._sess = tf.Session(config=tf.ConfigProto(
      log_device_placement=False,
      allow_soft_placement=True,
      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
        ))
    init = tf.global_variables_initializer()
    self._sess.run(init)
    self._sess.run(self._init_data_op)
    self._adam_varlist = filter(lambda var: 'Adam' not in var.name, tf.global_variables())

  def push_and_pull(self):
    if self._config.n_towers == 1:
      return 
    start = time.time()
    wrap_gradients = self._deltas + self._wrap_delta
    #self._sess.run(self._deltas)
    self._sess.run(wrap_gradients)
    # wrad 0.5 delta twice for better/faster convergence
    self._sess.run(self._wrap_delta)
    self._sess.run(self._sync_worker)
    elapsed = (time.time() - start)
    #self._reset_adam()
    logger.info("[push_and_pull] need:{} seconds to sync the model".format(elapsed))

  def _reset_adam(self):
    op_list = [var.assign(var.initialized_value()) for var in self._adam_varlist]
    self._sess.run(op_list)

  def run(self, feed_dict=None, test=False):
    #if self._config.n_towers > 1 and self.batch_count % self._config.commbatch == 0:
    if self.batch_count % self._config.commbatch == 0 and not test:
      logger.info('[run] batch_count:{}  sync the worker'.format(self.batch_count))
      #self.push_and_pull()
      t = threading.Thread(target=self.push_and_pull)
      t.start()
      logger.info('[run] batch_count:{}  sync the worker end'.format(self.batch_count))

    if not test:
      ret = self._sess.run(self._probs + self._labels + [self._work_train_op], feed_dict)
      self.batch_count += 1
      probs = ret[:self._config.n_towers]
      labels = ret[self._config.n_towers: self._config.n_towers * 2]
    else:
      ret = self._sess.run([self._probs[0], self._labels[0]], feed_dict=feed_dict)
      probs = ret[0]
      labels = ret[1]
    #logger.info("[run] batch_count:{}   commbatch:{}".format(self.batch_count, self._config.commbatch))
    #logger.info('all_variables:{}'.format(tf.all_variables()))
    return probs, labels
