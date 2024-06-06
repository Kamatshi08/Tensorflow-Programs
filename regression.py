import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)
X_train = np.random.rand(1000) * 100  
y_train = X_train * 3.5 + np.random.randn(1000) * 20  

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

model.compile(optimizer='adam', loss='mean_squared_error')


history = model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)


plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()


X_test = np.array([50, 100, 150], dtype=float)
y_pred = model.predict(X_test)

for i, size in enumerate(X_test):
    print(f"Predicted price for house of size {size} square meters: ${y_pred[i][0]:.2f}k")
