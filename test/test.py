
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
trained_model = tf.keras.models.load_model('../input/modelfile/returnfile.h5')
testinput = pd.read_csv('../input/modelfile/input.csv',header=None).to_numpy()
testinput = testinput.reshape((1,28,28))
print(trained_model.summary())
testoutput = trained_model.predict(testinput)
with open('modelsummary.txt','w') as ms:
    trained_model.summary(print_fn=lambda x: ms.write(x+'\n'))

with open('testoutput.txt','w') as testwriterfile:
    testwriterfile.write(str(testoutput))
    