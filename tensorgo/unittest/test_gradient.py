import tensorflow as tf
from tensorgo.utils import logger
import numpy as np

def test_record_variable():
  from tensorgo.tfutils.gradient import record_variable
  with tf.variable_scope('test_record_variable'):
    w = tf.get_variable('W', shape=[100, 1], dtype=tf.float32)
    b = tf.get_variable('b', shape=[1], dtype=tf.float32)
    varlist = [w, b]
  server_varlist = record_variable(varlist)
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  # assert length
  assert len(server_varlist) == len(varlist)
  # assert shape
  n = len(varlist)
  for i in range(n):
    varshape_0 = varlist[i].get_shape().as_list()
    varshape_1 = server_varlist[i].get_shape().as_list()
    assert varshape_0 == varshape_1
  logger.info('[test_record_variable] problem not found in test_record_variable')

def test_sync_variable():
  from tensorgo.tfutils.gradient import sync_variable
  from tensorgo.tfutils.gradient import record_variable
  # generate data
  with tf.variable_scope('test_sync_variable'):
    w = tf.get_variable('W', shape=[100, 1], dtype=tf.float32)
    b = tf.get_variable('b', shape=[1], dtype=tf.float32)
    varlist = [w, b]
  server_varlist = record_variable(varlist)
  sync_op = sync_variable(varlist, server_varlist)

  # run
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(sync_op)
  
  # check
  for i, var in enumerate(server_varlist):
    check_op = tf.equal(server_varlist[i], varlist[i])
    check_result = sess.run(check_op)
    assert np.mean(check_result) == 1

  logger.info('[test_sync_variable] problem not found in test_sync_variable')

if __name__ == '__main__':
  test_record_variable()
  test_sync_variable()
