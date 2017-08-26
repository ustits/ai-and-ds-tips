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


def modifiable_variable():
  x = tf.Variable(42)
  return tf.global_variables_initializer()
