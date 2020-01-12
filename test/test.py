
import tensorflow as tf
import numpy as np
import pandas as pd
trained_model = tf.keras.models.load_model('../input/modelfile/returnfile.h5')
testinput = pd.read_csv('../input/modelfile/input.csv',header=0).to_numpy()
testinput = testinput.reshape((28,28))
print(trained_model.summary())
print(trained_model.predict(testinput))
    