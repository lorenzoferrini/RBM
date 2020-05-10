import pickle
import gzip
import os
from sklearn.model_selection import StratifiedShuffleSplit

import numpy

def load_data( train_valid_ratio = 0.01):
  train_x_path = os.path.join(os.path.split(__file__)[0], "dataset/train-images-idx3-ubyte.gz")
  train_y_path = os.path.join(os.path.split(__file__)[0], "dataset/train-labels-idx1-ubyte.gz")
  test_x_path = os.path.join(os.path.split(__file__)[0], "dataset/t10k-images-idx3-ubyte.gz")
  test_y_path = os.path.join(os.path.split(__file__)[0], "dataset/t10k-labels-idx1-ubyte.gz")
  f1 = gzip.open(train_x_path, "rb")
  f2 = gzip.open(train_y_path, "rb")
  f3 = gzip.open(test_x_path, "rb")
  f4 = gzip.open(test_y_path, "rb")
  train_x = numpy.frombuffer(f1.read(), dtype=numpy.uint8).astype(numpy.float32)
  train_x = train_x[16:]/255
  print(train_x.size)
  train_x = train_x.reshape(int(train_x.size/(28*28)), 784)
  train_y = numpy.frombuffer(f2.read(), dtype=numpy.uint8).astype(numpy.float32)
  train_y = train_y[8:]
  test_x = numpy.frombuffer(f3.read(), dtype=numpy.uint8).astype(numpy.float32)
  test_x = test_x[16:]/255
  test_x = test_x.reshape(int(test_x.size/(28*28)), 784)
  test_y = numpy.frombuffer(f4.read(), dtype=numpy.uint8).astype(numpy.float32)
  test_y = test_y[8:]
  f1.close()
  f2.close()
  f3.close()
  f4.close()

  split = StratifiedShuffleSplit(n_splits=1, test_size=train_valid_ratio, random_state=42)
  for train_index, valid_index in split.split(train_x, train_y):
    train_set_x = train_x[train_index]
    train_set_y = train_y[train_index]
    valid_x = train_x[valid_index]
    valid_y = train_y[valid_index]

  rval = [(train_set_x, train_set_y), (valid_x, valid_y),
          (test_x, test_y)]

  return rval

