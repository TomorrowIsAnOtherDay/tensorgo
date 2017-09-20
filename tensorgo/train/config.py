#-*- coding: utf-8 -*-
#File: config.py
#Author: yobobobo(zhouboacmer@qq.com)

import tensorflow as tf
from tensorgo.utils import logger

__all__ = ['TrainConfig']

class TrainConfig(object):
  def __init__(self, dataset, model=None, n_towers=None, commbatch=50000):
    self.dataset = dataset
    self.model = model
    self.n_towers = n_towers
    self.commbatch = commbatch
