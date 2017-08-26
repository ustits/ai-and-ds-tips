import tensorflow as tf


def weights(n_features, n_labels):
  return tf.Variable(tf.truncated_normal((n_features, n_labels)))


def biases(n_labels):
  return tf.Variable(tf.zeros(n_labels))


def linear(x, w, b):
  """xW + b"""
  return tf.add(tf.matmul(x, w), b)
