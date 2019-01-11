import model_zoo as mz
import featurizer as ft
import numpy as np


featurize, weights = mz.load_80_1_model()


#print(sample)
for t in range(1, 24 * 10000):
    days = t / 2400.0
    sample = featurize(np.array([[days]]))
    result = weights.predict(sample)[0][0]

    print("{0:.3f}, {1}".format(days, result))
    if (result < .5): break