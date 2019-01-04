import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

def pad_zeros(width, x):
  return np.array(list(x) + [0] * (width - len(x)))

def array_map(f, x):
  return np.array(list(map(f, x)))

def array_map_2d(f, x):
  def ff(y):
     return np.array(list(map(f, y)))
  return np.array(list(map(ff, x)))

def sign(a):
  if (a > 0): return 1.0 
  else: return 0.0

def array_reverse(x):
  l = list(x)
  l.reverse()
  return l

def array_sign_interleave(x):
  abses = array_map(abs, x)
  signs = array_map(sign, x)
  return [val for pair in zip(abses, signs) for val in pair]

def remove_nans(a):
  return [x for x in a if str(x) != 'nan']

dataCsv = np.genfromtxt('data.csv', delimiter=',')

outputs, residue = np.hsplit(dataCsv, [1])
ratios, times = np.hsplit(residue, [1])
outputs = np.nan_to_num(outputs)
ratios = np.nan_to_num(ratios)
timeWindows = times.shape[1]

maxTime = max([max(map(max, times)), abs(min(map(min, times)))])
times = times / maxTime

def identity_transform(a): return a
def reverse_transform(a): 
  result = list(a)
  result.reverse()
  return result
def descending_transform(a): 
  return sorted(a.copy(), key = lambda x : -abs(x))
def ascending_transform(a): 
  return sorted(a.copy(), key = lambda x : abs(x))
def backfill(width, arr):
  x = arr.copy()
  total = x.shape[0]
  for i in range(0, width - 1):
    targetIndex = total - i - 1
    sourceIndex = width - i - 1
    if(x[targetIndex] == 0):
      x[targetIndex] = x[sourceIndex]
  return x


def imbed_reverse_expand_times(f):
  def ff(a):
    short = f(remove_nans(a))
    return array_sign_interleave(backfill(len(short), pad_zeros(timeWindows, short)))
  return array_map(ff, times)

def expand_times(f):
  def ff(a):
    short = f(remove_nans(a))
    return array_sign_interleave(pad_zeros(timeWindows, short))
  return array_map(ff, times)

forwardTimes = imbed_reverse_expand_times(identity_transform)
my_infile = np.column_stack((ratios, forwardTimes))
my_outfile = outputs
x_train, x_test, y_train, y_test = train_test_split(my_infile, my_outfile, test_size=0.01)
x_train.shape, y_train.shape, x_test.shape, y_test.shape


my_model = keras.Sequential([
    keras.layers.Dense(61, input_shape=(61,), kernel_initializer='normal', activation='sigmoid'),
    keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid')
])

my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

my_model.fit(x_train, y_train, epochs=1)
