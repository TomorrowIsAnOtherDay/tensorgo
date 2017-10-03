#-*- coding: utf-8 -*-
#File: multigpu.py
#Author: yobobobo(zhouboacmer@qq.com)


import numpy as np
import tensorflow as tf
from tensorgo.utils import logger
from tensorgo.train.config import TrainConfig
from tensorgo.tfutils.gradient import record_variable, update_variable, \
    compute_worker_gradient, apply_worker_gradients, sync_variable, fetch_all_vars
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
    self._worker_param = []
    self._deltas = []
    self._wrap_delta = []
    self._sync_worker = []

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
            loss = self._model.build_graph(self._data_iter)
            varlist = tf.trainable_variables()
            val_keyword = 'tower_{}'.format(i)
            varlist = filter(lambda x: val_keyword in x.name, varlist)
            self._worker_param.append(varlist)

            # record server model
            if i == 0:
              with tf.device('/cpu:0'):
                self._server_param = record_variable(varlist)

            #push & pull
            sync_worker = update_variable(self._server_param, varlist)
            with tf.device('/cpu:0'):
              delta = compute_worker_gradient(self._server_param, varlist)
              wrap_delta = apply_worker_gradients(opt, self._server_param, delta, scale=(1.0 / self._config.n_towers) * 0.5)
            self._sync_worker.extend(sync_worker)
            self._deltas.extend(delta)
            self._wrap_delta.append(wrap_delta)
            # end
            worker_train_op = opt.minimize(loss)
            self._work_train_op.append(worker_train_op)

    # init session
    self._sess = tf.Session(config=tf.ConfigProto(
      log_device_placement=False,
      allow_soft_placement=True,
      gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
        ))
    
    # init graph
    self.init_graph()

    # init input
    self._sess.run(self._init_data_op)
    self._adam_varlist = filter(lambda var: 'Adam' not in var.name, tf.global_variables())

  def init_graph(self):
    """init woker and server variable, all the variable should be same after initialization"""
    logger.info("Initializing global parameters")
    init = tf.global_variables_initializer()
    self._sess.run(init)

    # used first woker parameter as global initial parameter
    return 
    for i in range(1, self._config.n_towers):
      sync_op = sync_variable(self._worker_param[0], self._worker_param[i])
      self._sess.run(sync_op)
    sync_server_op = sync_variable(self._worker_param[0], self._server_param)
    self._sess.run(sync_server_op)
    logger.info("Global parameters, all parameter are same")

  def push_and_pull(self):
    """aggregate workers' gradient to server and sync the model of workers"""

    # unnecessary to sync model when there is only one worker
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

  def run(self, fetches=[], feed_dict=None, test=False, one_worker=False):
    """
    Args:
        fetches: A single graph element, a list of graph elements
        feed_dict:  A dictionary that maps graph elements to values
        test: whether to run train op
        one_worker: use only one worker(often used in evaluation, which could keep data's order )
    """
    if self.batch_count % self._config.commbatch == 0 and not test:
      logger.info('[run] batch_count:{}  sync the worker'.format(self.batch_count))
      #self.push_and_pull()
      t = threading.Thread(target=self.push_and_pull)
      t.start()
      logger.info('[run] batch_count:{}  sync the worker end'.format(self.batch_count))

    if one_worker == False:
      final_fetches = fetch_all_vars(self._worker_param, fetches, self._config.n_towers)
    else: 
      final_fetches = fetches

    if not test:
      final_fetches.append(self._work_train_op)
    self.batch_count += 1

    ret = self._sess.run(final_fetches, feed_dict=feed_dict)
    # reorganize its output
    if len(fetches) == 0:
      return []
    else:
      merge_ret = []
      fetches_num = len(fetches)

      for i in range(fetches_num):
        if one_worker:
          merge_ret.append(ret[i])
        else:
          merge_ret.append(np.concatenate(ret[i]))
      return merge_ret
