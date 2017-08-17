import matplotlib.pyplot as plt
import random
import numpy as np


def sum_of_squares(v):
  return sum(v_i ** 2 for v_i in v)


# отношение приращения
def difference_quitient(f, x, h):
  return (f(x + h) - f(x)) / h


def square(x):
  return x * x


def derivative(x):
  return 2 * x


derivative_estimate = lambda x: difference_quitient(square, x, h=0.00001)

xs = range(-10, 10)
plt.title("Фактические производные и их оценки в сравнении")
plt.plot(xs, list(map(derivative, xs)), 'rx')
plt.plot(xs, list(map(derivative_estimate, xs)), 'b+')
plt.show()


def partial_difference_quotient(f, v, i, h):
  """ добавляет h только когда j == i """
  w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
  return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.00001):
  return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]


def step(v, direction, step_size):
  return [v_i + step_size * direction_i for v_i, direction_i in zip(v, direction)]


def distance(v1, v2):
  return np.linalg.norm(np.array(v1) - np.array(v2))


def gradient_fn():
  def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]

  """ выбираем случайную точку в трехмерном пространстве """
  v = [random.randint(-10, 10) for i in range(3)]

  """ точность расчета """
  tolerance = 0.0000001

  while True:
    gradient = sum_of_squares_gradient(v)
    next_v = step(v, gradient, -0.01)
    if distance(v, next_v) < tolerance:
      break
    v = next_v

  print(v)


# результат всегда будет близок к (0, 0, 0)
gradient_fn()


def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
  def safe(f):

    def safe_f(*args, **kwargs):
      try:
        return f(*args, **kwargs)
      except:
        return float('inf')

    return safe_f

  step_sizes = [100 / 10 ** i for i in range(8)]

  theta = theta_0
  target_fn = safe(target_fn)

  value = target_fn(theta)

  while True:
    gradient = gradient_fn(theta)
    next_thetas = [step(theta, gradient, -step_size)
                   for step_size in step_sizes]

    """найдем значение theta при котором получаем минимальное значения для функции target_fn"""
    next_theta = min(next_thetas, key=target_fn)
    next_value = target_fn(next_theta)
    if distance(value, next_value) < tolerance:
      return theta
    else:
      theta, value = next_theta, next_value


def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
  def negate(f):
    return lambda *args, **kwargs: -f(*args, **kwargs)

  def negate_all(f):
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]

  return minimize_batch(negate(target_fn),
                        negate_all(gradient_fn),
                        theta_0,
                        tolerance)


def in_random_order(data):
  indexes = [i for i, _ in enumerate(data)]
  random.shuffle(indexes)
  for i in indexes:
    yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):

  def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]

  def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

  data = zip(x, y)
  theta = theta_0
  alpha = alpha_0
  min_theta, min_value = None, float("inf")
  iterations_with_no_improvement = 0

  while iterations_with_no_improvement < 100:
    value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

    if value < min_value:
      min_theta, min_value = theta, value
      iterations_with_no_improvement = 0
      alpha = alpha_0
    else:
      iterations_with_no_improvement += 1
      alpha *= 0.9

      for x_i, y_i in in_random_order(data):
        gradient_i = gradient_fn(x_i, y_i, theta)
        theta = vector_subtract(theta,
                                scalar_multiply(alpha, gradient_i))

  return min_theta