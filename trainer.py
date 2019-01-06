# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from keras.models import load_model
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio
from scipy.special import logit
from scipy.special import expit
from sklearn.preprocessing import scale
from os import walk
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

print(tf.__version__)


from numpy import genfromtxt

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

dataCsv = genfromtxt('data.csv', delimiter=',')

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

def color_of_answer(answer):
  if (answer > 0.5): return 'blue'
  else: return 'gray'

forwardTimes = imbed_reverse_expand_times(identity_transform)
my_infile = np.column_stack((ratios, forwardTimes))
my_outfile = outputs
x_train, x_test, y_train, y_test = train_test_split(my_infile, my_outfile, test_size=0.2)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

model_file = "60-10-1.h5"

my_model = keras.Sequential([
    keras.layers.Dense(60, input_shape=(61,), kernel_initializer='normal', activation='sigmoid'),
    keras.layers.Dense(10, kernel_initializer='normal', activation='sigmoid'),
    keras.layers.Dense(1, kernel_initializer='normal', activation='sigmoid')
])

my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
total_epochs = 0
total_steps = 0
x_disp = x_test
y_disp = y_test
tpy = np.transpose(y_disp)[0]
c_disp = array_map(color_of_answer, tpy)

noise1 = np.random.normal(0, 0.5, y_disp.shape[0])
noise2 = np.random.normal(0, 0.5, y_disp.shape[0])

for filename in sorted(os.listdir(".")):
  if (model_file in filename and ".png" in filename):
    os.remove(filename)
    
    
class TrainState:
  total_steps = 0
  prior_loss = 1.00
  latest_steps = 1
  latest_loss = 1.00

state = TrainState()

goal_loss = 0.7
final_goal_loss = 0.2
loss_step = -0.005
render_countdown = 5



gif_file = "{}.gif".format(model_file)

def train_steps(steps, state):
  if (steps < state.latest_steps): steps = state.latest_steps
  state.prior_loss = state.latest_loss
  #print("Stepping {} starting at {}".format(steps, state.total_steps))
  my_model.fit(x_train, y_train, epochs=1, steps_per_epoch=steps, verbose = 0)
  loss, acc = my_model.evaluate(x_test, y_test, verbose=0)
  print("- step {0} loss = {1:.4f} accuracy = {2:.4f}".format(state.total_steps, loss, acc))
  state.latest_loss = loss
  state.total_steps = state.total_steps + steps
  return state

def train_until(target_loss, state):
  print("{0}: Seeking target loss {1:.4f}".format(render_countdown, target_loss))
  while True: 
    target_loss_delta = 0.7 - state.latest_loss
    estimated_loss_per_step = target_loss_delta / (1 + state.total_steps)

    if (estimated_loss_per_step > 0):
      steps_needed = int((state.latest_loss - target_loss) / estimated_loss_per_step)
      train_steps(steps_needed, state)                              
    else:
      train_steps(state.latest_steps, state)   
      
    if (target_loss > state.latest_loss): 
      print("- met loss goal of {0:.4f} at step {1} with actual loss {0:.4f}".format(target_loss, state.total_steps, state.latest_loss))
      return state
        
def save_image(state):
    predictions = my_model.predict(x_disp)
    loss, acc = my_model.evaluate(x_disp, y_disp, verbose=0)

    tpp = np.transpose(predictions)[0]
    n1 = noise1 * loss
    n2 = noise2 * loss
    sns.set_style("dark")
    plt.clf()
    plt.ylim(-0.5, 1.5)
    plt.xlim(-0.5, 1.5)
    title = "{0}: Step {1} loss={2:.4f} accuracy={3:.4f}".format(model_file, state.total_steps, loss, acc)
    plt.title(title)

    xpn = tpp + n1
    ypn = 0.5 + n2

    plt.scatter(xpn, ypn, color=c_disp)

    image_name = "{}_{}.png".format(model_file, 100000 + state.total_steps)
    plt.savefig(image_name)  

def render_gif():
    with imageio.get_writer(gif_file, mode='I', fps = 10) as writer:
      for filename in sorted(os.listdir(".")):
        if (model_file in filename and ".png" in filename):
          writer.append_data(imageio.imread(filename))
  
while True:
  if (goal_loss <= final_goal_loss): break
  state = train_until(goal_loss, state)
  # save_image(state)
  goal_loss = goal_loss + loss_step
  render_countdown = render_countdown - 1
  if (render_countdown == 0):
    render_countdown = 5
    my_model.save(model_file)
    render_gif()

