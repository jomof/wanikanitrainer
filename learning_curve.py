import model_zoo as mz
import featurizer as ft
import numpy as np


featurize, weights = mz.load_70_1_model()


#print(sample)
for t in range(1, 24 * 5):
    days = t / 24.0
    sample = featurize(np.array([[days, -1.0, 1.0, -1.0, -1.0]]))
    result = weights.predict(sample)
    print("{0:.3f}, {1}".format(days, result[0]))