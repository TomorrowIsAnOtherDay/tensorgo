#-*- coding: utf-8 -*-
#File: gradient.py
#Author: yobobobo(zhouboacmer@qq.com)
import tensorflow as tf

__all__ = ['record_variable']

def record_variable(varlist):
  """ record variable list as parameter server 
    @param:
      varlist: variable list which should sync among workers and servers
  """
  record_vars = []
  for var in varlist:
    var_name = '_'.join(var.name.split('/')[-3:])
    var_name = var_name.replace(':', '_')
    record_var = tf.get_variable('server_{}'.format(var_name), shape=var.get_shape(), trainable=False)
    record_vars.append(record_var)
  return record_vars

def update_variable(server_varlist, worker_varlist):
  assert len(worker_varlist) == len(server_varlist)
  n_vars = len(worker_varlist)
  updated_vals = []
  for i in xrange(n_vars):
    updated_val = worker_varlist[i].assign_add(server_varlist[i] - worker_varlist[i])
    updated_vals.append(updated_val)
  return updated_vals

def compute_worker_gradient(server_varlist, worker_varlist):
  assert len(server_varlist) == len(worker_varlist)
  n_vars = len(server_varlist)
  deltas = []
  for i in xrange(n_vars):
    delta = server_varlist[i] - worker_varlist[i]
    deltas.append(delta)
  return deltas

def apply_worker_gradients(opt, server_param, delta, scale=1.0):
  assert len(server_param) == len(delta)
  var_grads = []
  n_vars = len(server_param)
  for i in xrange(n_vars):
    var_grads.append((delta[i], server_param[i]))
  apply_gradients = opt.apply_gradients(var_grads)
  return apply_gradients
