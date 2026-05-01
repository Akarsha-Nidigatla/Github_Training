import tensorflow as tf

# Display TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Create a simple constant tensor
hello = tf.constant('Hello, TensorFlow!')
print(hello)

# Create a simple math operation
a = tf.constant(5)
b = tf.constant(3)
print(f"Addition: {a + b}")

# Neural network example - simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(3, activation='softmax')
])

print("\nModel Summary:")
model.summary()

# Generate sample data
import numpy as np
X = np.random.randn(10, 4)
y = np.random.randint(0, 3, 10)

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nTraining the model...")
history = model.fit(X, y, epochs=3, verbose=1)

print("\nTensorFlow sample completed successfully!")