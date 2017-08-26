import tensorflow as tf


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


def softmax():
  logit_data = [2.0, 1.0, 0.1]
  logits = tf.placeholder(tf.float32)

  softmax = tf.nn.softmax(logits)

  return tf.Session().run(softmax, feed_dict={logits: logit_data})


def cross_entropy():
  softmax_data = [0.7, 0.2, 0.1]
  one_hot_data = [1.0, 0.0, 0.0]

  softmax = tf.placeholder(tf.float32)
  one_hot = tf.placeholder(tf.float32)

  cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
  return tf.Session().run(cross_entropy, feed_dict={softmax: softmax_data,
                                                    one_hot: one_hot_data})
