import numpy as np
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def init_params(layer_dims):
    np.random.seed(43)
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params["b" + str(l)] = np.zeros((layer_dims[l], 1))

    return params


# Z (linear hypothesis) - Z = W*X + b ,
# W - weight matrix, b- bias vector, X- Input
def sigmoid(Z):
    A = 1 / (1 + np.exp(np.dot(-1, Z)))
    cache = Z

    return A, cache


def forward_prop(X, params):

    A = X  # input to first layer i.e. training data
    caches = []
    L = len(params) // 2
    for l in range(1, L + 1):
        A_prev = A

        # Linear Hypothesis
        Z = np.dot(params["W" + str(l)], A_prev) + params["b" + str(l)]

        # Storing the linear cache
        linear_cache = (A_prev, params["W" + str(l)], params["b" + str(l)])

        # Applying sigmoid on linear hypothesis
        A, activation_cache = sigmoid(Z)

        # storing the both linear and activation cache
        cache = (linear_cache, activation_cache)
        caches.append(cache)

    return A, caches


def cost_function(A, Y):
    m = Y.shape[1]

    cost = (-1 / m) * (np.dot(np.log(A), Y.T) + np.dot(np.log(1 - A), 1 - Y.T))

    return cost


def one_layer_backward(dA, cache):
    linear_cache, activation_cache = cache

    Z = activation_cache
    A, _ = sigmoid(Z)
    dZ = dA * A * (1 - A)

    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def backprop(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # Derivative of cost w.r.t. final activation
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Backprop for output layer (layer L)
    current_cache = caches[-1]
    dA_prev, dW, db = one_layer_backward(dAL, current_cache)
    grads["dA" + str(L)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db

    # Loop over previous layers
    for l in reversed(range(1, L)):
        current_cache = caches[l - 1]
        dA_prev, dW, db = one_layer_backward(grads["dA" + str(l + 1)], current_cache)
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l)] = dW
        grads["db" + str(l)] = db

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters


def train(X, Y, layer_dims, epochs, lr):
    params = init_params(layer_dims)
    cost_history = []

    for i in range(epochs):
        Y_hat, caches = forward_prop(X, params)
        cost = cost_function(Y_hat, Y)
        cost_history.append(cost)
        grads = backprop(Y_hat, Y, caches)

        params = update_parameters(params, grads, lr)

    return params, cost_history


def predict(X, params):
    A, _ = forward_prop(X, params)
    predictions = (A > 0.5).astype(int)
    return predictions


# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# store data as a Bernoulli variable
y_binary = (y > 0).astype(int)

# clean data by removing any NaN values
X_clean = X.dropna()
y_binary = y_binary.loc[X_clean.index]

# scale the data into a standardized form
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)
print(X_scaled.shape)

# split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_binary, test_size=0.2, random_state=42
)

# shape data for input
X_train = X_train.T  # Shape: (13, 237)
X_test = X_test.T  # Shape: (13, 60)

y_train = y_train.to_numpy().reshape(1, -1)  # Shape: (1, 237)
y_test = y_test.to_numpy().reshape(1, -1)  # Shape: (1, 60)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# train model
layer_dims = [13, 3, 1]  # array of layers
learning_rate = 0.01  # Stable training
epochs = 6000  # short-length run
params, cost_history = train(X_train, y_train, layer_dims, epochs, learning_rate)

# analyze model performance
y_pred_test = predict(X_test, params)
accuracy = np.mean(y_pred_test == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# print test cases
preds = y_pred_test.flatten()
labels = y_test.flatten()
print("Index | Predicted | Actual")
print("---------------------------")
for i, (pred, true) in enumerate(zip(preds, labels)):
    print(f"{i:5} | {pred:9} | {true}")

# confusion matrix
preds = predict(X_test, params).flatten()
true = y_test.flatten()
cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Disease", "Disease"]
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

W1 = params["W1"]  # Shape: (hidden layer, input layer)
b1 = params["b1"]
feature_names = X.columns
weights_df = pd.DataFrame(
    W1, columns=feature_names, index=["Neuron 1", "Neuron 2", "Neuron 3"]
)
print(weights_df.T)

W2 = params["W2"]  # Shape: (output layer, hidden layer)
print("Output layer weights (W2):", W2)
