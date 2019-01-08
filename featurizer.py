import numpy as np

# Pad an array so that it's width is equal to width
def pad_zeros(width, x): return np.array(list(x) + [0] * (width - len(x)))

# Map an array with a function
def array_map(f, x): return np.array(list(map(f, x)))

# Convert negative number to 1.0 and positive number to 0.0
def sign_to_bool(a):
  if (a > 0): return 1.0 
  else: return 0.0

# Remove nan values from the array
def remove_nans(a): return [x for x in a if str(x) != 'nan']

# Given an array with positive and negative floats replace with
# absolute value followed by 1.0 (for positive) or 0.0 (for negative)
def array_sign_interleave(x):
  abses = array_map(abs, x)
  signs = array_map(sign_to_bool, x)
  return [val for pair in zip(abses, signs) for val in pair]


# Given an array padded on the right with zeros then fill the zeros with
# values shifted from left.
# [4,3,2,1,0,0,0] becomes [4,3,2,1,3,2,1]
def backfill(width, arr):
  x = arr.copy()
  total = x.shape[0]
  for i in range(0, width - 1):
    targetIndex = total - i - 1
    sourceIndex = width - i - 1
    if(x[targetIndex] == 0):
      x[targetIndex] = x[sourceIndex]
  return x

def imbed_reverse_expand_times(f, times, timeWindows):
  def ff(a):
    short = f(remove_nans(a))
    return array_sign_interleave(backfill(len(short), pad_zeros(timeWindows, short)))
  return array_map(ff, times)

def expand_times(f, times, timeWindows):
  def ff(a):
    short = f(remove_nans(a))
    return array_sign_interleave(pad_zeros(timeWindows, short))
  return array_map(ff, times)

def identity_transform(a): return a
def reverse_transform(a): 
  result = list(a)
  result.reverse()
  return result
def descending_transform_abs(a): return sorted(list(a), key = lambda x : -abs(x))
def ascending_transform_abs(a): return sorted(list(a), key = lambda x : abs(x))
def descending_transform(a): return sorted(list(a), key = lambda x : -x)
def ascending_transform(a): return sorted(list(a), key = lambda x : x)

def featurize(dataCsv):
  outputs, residue = np.hsplit(dataCsv, [1])
  ratios, times = np.hsplit(residue, [1])
  outputs = np.nan_to_num(outputs)
  ratios = np.nan_to_num(ratios)
  timeWindows = times.shape[1]

  maxTime = max([max(map(max, times)), abs(min(map(min, times)))])
  times = times / maxTime

  forwardTimes = imbed_reverse_expand_times(identity_transform, times, timeWindows)
  reverseTimes = imbed_reverse_expand_times(reverse_transform, times, timeWindows)
  descendingTimesAbs = imbed_reverse_expand_times(descending_transform_abs, times, timeWindows)
  ascendingTimesAbs = imbed_reverse_expand_times(ascending_transform_abs, times, timeWindows)
  descendingTimes = imbed_reverse_expand_times(descending_transform, times, timeWindows)
  ascendingTimes = imbed_reverse_expand_times(ascending_transform, times, timeWindows)

  inputs = np.column_stack((forwardTimes, reverseTimes, descendingTimesAbs, ascendingTimesAbs, descendingTimes, ascendingTimes))
  return inputs, outputs

# Tests
def test_all_featurizer():
  array = np.array([1,2,3])
  padded = pad_zeros(5, array)
  if (padded.shape[0] != 5): raise Exception("wrong size")
  if (padded[3] != 0): raise Exception("expected zero")

  negated = array_map(lambda x: -x, array)
  if (negated.shape[0] != 3): raise Exception("wrong size")
  if (negated[1] != -2): raise Exception("wrong value {}".format(negated))



test_all_featurizer()