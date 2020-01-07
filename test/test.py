
import tensorflow as tf
trained_model = tf.keras.models.load_model('../input/returnfile.h5')
print(trained_model.summary())
    