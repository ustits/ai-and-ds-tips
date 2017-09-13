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
  two_dims = tf.placeholder(tf.int32, [None, None])
  output = tf.Session().run(two_dims, feed_dict={string_type: 'Hello world', int_type: 123,
                                                 float_type: 32.43, two_dims: [[1, 2], [3, 4]]})
  print(output)


def modifiable_variable():
  x = tf.Variable(42)
  run_variable(x)


def variable_generator():
  shape = (5, 3)
  result_from = -1
  result_to = 1
  x = tf.Variable(tf.random_uniform(shape, result_from, result_to))
  run_variable(x)


def run_variable(x):
  sess = tf.Session()
  # must initialize variables at first
  sess.run(tf.global_variables_initializer())
  result = sess.run(x)
  print(result)
