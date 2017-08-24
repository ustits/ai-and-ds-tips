import tensorflow as tf


def run_session(tensor):
  output = tf.Session().run(tensor)
  print(output)


def simple():
  """defining a simple tensor"""
  return tf.constant('Hello world!')


def work_with_non_constant():
  string_type = tf.placeholder(tf.string)
  int_type = tf.placeholder(tf.int32)
  float_type = tf.placeholder(tf.float32)
  output = tf.Session().run(string_type, feed_dict={string_type: 'Hello world', int_type: 123,
                                                    float_type: 32.43})
  print(output)


def math():
  """take two tensor, apply operation, get another tensor"""
  add = tf.add(5, 2)
  subtract = tf.subtract(7, 3)
  multiply = tf.multiply(4, 3)
  return add, subtract, multiply


def some_complex_math():
  x = tf.constant(10)
  y = tf.constant(2)
  z = tf.subtract(tf.divide(x, y), 1)
  return z


def converting_types():
  return tf.add(tf.constant(1), tf.cast(tf.constant(2.0), tf.int32))
