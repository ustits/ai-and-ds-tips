import matplotlib.pyplot as plt
import numpy as np


def simple_plot():
  plt.plot([1, 2, 3, 4])
  plt.ylabel('some numbers')
  plt.show()


def simple_plot2():
  plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
  plt.ylabel('some numbers')
  plt.show()


def func_plot():
  """
  `r--` - red dashes
  `bs` - blue squares
  `g^` - green triangles
  https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot - other markers
  """
  t = np.arange(0, 5, 0.2)
  plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
  plt.show()


def func_plot2():
  def func(x):
    return 64 - 16 * (x - 1) ** 2

  t = np.arange(0, 4, 0.1)
  plt.plot(t, func(t))
  plt.grid(True)
  # draw axis
  plt.axhline(y=0, color='k')
  plt.axvline(x=0, color='k')
  plt.show()


def single_point():
  plt.plot(1, marker='o', markersize=3, color='red')
  plt.show()

