import tensorflow as tf
import featurizer as ft

# Accuracy 89%
# model = keras.Sequential([
#     keras.layers.Dense(70, input_shape=(60,), activation='sigmoid'),
#     keras.layers.Dense(1, activation='relu')
# ])
def load_70_1_model():
    # The featurizer used for this model
    def featurize(times):
        maxTime = 359.122
        timeWindows = 30
        times = times / maxTime
        return ft.imbed_reverse_expand_times(ft.identity_transform, times, timeWindows)
    return featurize, tf.keras.models.load_model("70-1-complete.h5")

# Accuracy 89%
# model = keras.Sequential([
#     keras.layers.Dense(80, input_shape=(60,), activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')
# ])
def load_80_1_model():
    # The featurizer used for this model
    def featurize(times):
        maxTime = 359.122
        timeWindows = 30
        times = times / maxTime
        return ft.imbed_reverse_expand_times(ft.identity_transform, times, timeWindows)
    return featurize, tf.keras.models.load_model("80-1-complete.h5")




