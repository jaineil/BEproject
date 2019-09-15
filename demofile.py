
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense([100], input_shape=[16], activation=tf.nn.relu),
        tf.keras.layers.Dense([2], activation=tf.nn.softmax),])
model.compile(optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])