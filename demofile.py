
import tensorflow as tf
dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28))
        ,tf.keras.layers.Dense(128, activation=tf.nn.relu)
        ,tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test,  y_test, verbose=2)
model.save('returnfile.h5')