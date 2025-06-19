import numpy as np
import os
import matplotlib.pyplot as plt
from tensor import Tensor
import ipywidgets as widgets
from IPython.display import display, clear_output

file_path = "mnist.npz"

try:
    # Open the .npz file
    with np.load(file_path, allow_pickle=True) as data:
        # Split training data: first 50000 for training, last 10000 for validation
        x_train = data['x_train'][:50000]  # First 50000 training images
        x_dev = data['x_train'][50000:]    # Last 10000 as validation images
        y_train = np.eye(10)[data['y_train'][:50000]]  # First 50000 training labels
        y_dev = np.eye(10)[data['y_train'][50000:]]    # Last 10000 as validation labels
        x_test = data['x_test']            # Test images
        y_test = np.eye(10)[data['y_test']]            # Test labels

        print("MNIST data loaded successfully!")
        print(f"Training images shape: {x_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Validation images shape: {x_dev.shape}")
        print(f"Validation labels shape: {y_dev.shape}")
        print(f"Test images shape: {x_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")

#initialize the parameters
batch_size = 30
max_steps = 10000
n_hidden = 100
n_input = 28*28
n_output = 10
learning_rate = 0.01
loss_history = []

#initialize the weights and biases
w1 = Tensor(np.random.randn(n_input, n_hidden) * (1/np.sqrt(n_input)))
b1 = Tensor(np.zeros(n_hidden))
w2 = Tensor(np.random.randn(n_hidden, n_output)* (1/np.sqrt(n_hidden)))
b2 = Tensor(np.zeros(n_output))

for i in range(max_steps):
    #create a batch of data
    batch_idx = np.random.randint(0, x_train.shape[0], batch_size)
    x_batch = Tensor(x_train[batch_idx].reshape(-1, 28*28).astype(np.float64)/255)
    y_batch = Tensor(y_train[batch_idx].astype(np.float64))

    #forward passx
    h1 = x_batch @ w1 + b1
    h1t = h1.tanh()
    h2 = h1t @ w2 + b2
    loss = h2.cross_entropy(y_batch.data)

    #backward pass
    w1.grad.fill(0)
    b1.grad.fill(0)
    w2.grad.fill(0)
    b2.grad.fill(0)
    loss.backward()

    #add the loss to the loss history
    loss_history.append(loss.data.mean())

    #update the weights and biases
    w1.data -= w1.grad * learning_rate
    b1.data -= b1.grad * learning_rate
    w2.data -= w2.grad * learning_rate
    b2.data -= b2.grad * learning_rate
    if i % 100 == 0:
        print(f"Step {i} Loss: {loss.data.mean()}")

loss_array = np.array(loss_history)

averaging_step = 100

# Trim the array so its length is a multiple of the step size
end_index = len(loss_array) - (len(loss_array) % averaging_step)
trimmed_array = loss_array[:end_index]

# Reshape the 1D array into a 2D array (e.g., 2000 -> 20x100) and take the mean of each row
averaged_losses = trimmed_array.reshape(-1, averaging_step).mean(axis=1)

plt.plot(averaged_losses)
plt.show()


x_batch = Tensor(x_train.reshape(-1, 28*28).astype(np.float64))
y_batch = Tensor(y_train.astype(np.float64))

h1 = x_batch @ w1 + b1
h1t = h1.tanh()
h2 = h1t @ w2 + b2
loss = h2.cross_entropy(y_batch.data)

predicted_labels = np.argmax(h2.data, axis=1)
correct_predictions = (predicted_labels == np.argmax(y_batch.data, axis=1))
print("Train accuracy: ", np.mean(correct_predictions) * 100)
print("Train loss: ", loss.data)

x_batch = Tensor(x_dev.reshape(-1, 28*28).astype(np.float64))
y_batch = Tensor(y_dev.astype(np.float64))

h1 = x_batch @ w1 + b1
h1t = h1.tanh()
h2 = h1t @ w2 + b2
loss = h2.cross_entropy(y_batch.data)

predicted_labels = np.argmax(h2.data, axis=1)
correct_predictions = (predicted_labels == np.argmax(y_batch.data, axis=1))
print("Val accuracy: ", np.mean(correct_predictions) * 100)
print("Val loss: ", loss.data)
