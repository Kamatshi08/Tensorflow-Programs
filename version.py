import tensorflow as tf
import numpy as np


x_train = np.linspace(-1, 1, 100)
y_train = 2 * x_train + np.random.randn(*x_train.shape) * 0.3  

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])  
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=100)

x_test = np.array([2.0, 3.0, 4.0])
y_pred = model.predict(x_test)
print("Predictions:", y_pred.flatten())
