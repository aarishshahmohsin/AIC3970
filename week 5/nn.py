import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('obesity_data.csv')
np.random.seed(42)

label_encoder = LabelEncoder()
data['ObesityCategory'] = label_encoder.fit_transform(data['ObesityCategory'])

X = data.drop(columns=['ObesityCategory']).values
y = data['ObesityCategory'].values.reshape(-1, 1)  

# gender (0, 1, female, male)
X[:, 1] = np.where(X[:, 1] == 'Male', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class NeuralNetwork:
    def __init__(self, layers, loss_function, dropout_rate=0.2):
        self.layers = layers
        self.loss_function = loss_function
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.train_loss_history = []
        self.val_loss_history = []
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i-1], layers[i]) * 0.01)
            self.biases.append(np.zeros((1, layers[i])))

    def forward(self, X, training=True):
        self.layer_outputs = []
        self.dropout_masks = []
        input_data = X
        for i in range(len(self.weights)):
            z = np.dot(input_data, self.weights[i]) + self.biases[i]
            if i == len(self.weights) - 1:  
                a = sigmoid(z)
            else:  
                a = relu(z)
                if training:  
                    mask = (np.random.rand(*a.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                    a *= mask
                    self.dropout_masks.append(mask)
            self.layer_outputs.append(a)
            input_data = a
        return input_data

    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        y_pred = self.layer_outputs[-1]
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        if self.loss_function == 'mse':
            d_loss = (y_pred - y) / m
        elif self.loss_function == 'cross_entropy':
            d_loss = (y_pred - y) / (y_pred * (1 - y_pred)) / m

        for i in reversed(range(len(self.weights))):
            a_prev = self.layer_outputs[i-1] if i > 0 else X
            dW = np.dot(a_prev.T, d_loss)
            db = np.sum(d_loss, axis=0, keepdims=True)
            if i > 0:  # Not the first layer
                d_loss = np.dot(d_loss, self.weights[i].T) * relu_derivative(a_prev)
                if self.dropout_masks:  # Apply dropout mask
                    d_loss *= self.dropout_masks[i-1]
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * db

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=40, learning_rate=0.01, early_stopping=False, patience=10):
        best_loss = np.inf
        best_weights = None
        best_biases = None
        no_improvement = 0

        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            for start in range(0, X_train.shape[0], batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]
                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                self.forward(X_batch, training=True)
                self.backward(X_batch, y_batch, learning_rate)

            y_train_pred = self.forward(X_train, training=False)
            if self.loss_function == 'mse':
                train_loss = mean_squared_error(y_train, y_train_pred)
            elif self.loss_function == 'cross_entropy':
                train_loss = cross_entropy_loss(y_train, y_train_pred)
            self.train_loss_history.append(train_loss)

            y_val_pred = self.forward(X_val, training=False)
            if self.loss_function == 'mse':
                val_loss = mean_squared_error(y_val, y_val_pred)
            elif self.loss_function == 'cross_entropy':
                val_loss = cross_entropy_loss(y_val, y_val_pred)
            self.val_loss_history.append(val_loss)

            if early_stopping:
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_weights = [w.copy() for w in self.weights]
                    best_biases = [b.copy() for b in self.biases]
                    no_improvement = 0
                else:
                    no_improvement += 1
                    if no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

            print(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        if early_stopping:
            self.weights = best_weights
            self.biases = best_biases

    def evaluate(self, X_test, y_test):
        y_pred = self.forward(X_test, training=False)
        if self.loss_function == 'mse':
            test_loss = mean_squared_error(y_test, y_pred)
        elif self.loss_function == 'cross_entropy':
            test_loss = cross_entropy_loss(y_test, y_pred)
        accuracy = np.mean((y_pred > 0.5) == y_test)
        return test_loss, accuracy

n1_layers = [X_train.shape[1], 64, 32, 16, 1]  # N1: 3 hidden layers
n2_layers = [X_train.shape[1], 128, 64, 32, 16, 1]  # N2: 4 hidden layers


select = input("select between (cross_entropy (ce)), (mse (mse))")

if select == 'ce':
    print("Training N1 with CrossEntropy (without early stopping)...")
    n1_ce_no_early_stop = NeuralNetwork(n1_layers, loss_function='cross_entropy')
    n1_ce_no_early_stop.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=40, learning_rate=0.01, early_stopping=False)
    test_loss, accuracy = n1_ce_no_early_stop.evaluate(X_test, y_test)
    print(f"N1 with CrossEntropy (without early stopping) - Test Loss: {test_loss}, Test Accuracy: {accuracy}")

    print("Training N1 with CrossEntropy (with early stopping)...")
    n1_ce_early_stop = NeuralNetwork(n1_layers, loss_function='cross_entropy')
    n1_ce_early_stop.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=40, learning_rate=0.01, early_stopping=True, patience=10)
    test_loss, accuracy = n1_ce_early_stop.evaluate(X_test, y_test)
    print(f"N1 with CrossEntropy (with early stopping) - Test Loss: {test_loss}, Test Accuracy: {accuracy}")

    plt.figure(figsize=(12, 6))
    plt.plot(n1_ce_no_early_stop.train_loss_history, label='Training Loss (No Early Stopping)')
    plt.plot(n1_ce_no_early_stop.val_loss_history, label='Validation Loss (No Early Stopping)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('N1 with Cross Entropy - Loss History (Without Early Stopping)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(n1_ce_early_stop.train_loss_history, label='Training Loss (With Early Stopping)')
    plt.plot(n1_ce_early_stop.val_loss_history, label='Validation Loss (With Early Stopping)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('N1 with Cross Entropy - Loss History (With Early Stopping)')
    plt.legend()
    plt.show()

elif select == 'mse':
    print("Training N1 with MSE (without early stopping)...")
    n1_mse_no_early_stop = NeuralNetwork(n1_layers, loss_function='mse')
    n1_mse_no_early_stop.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=40, learning_rate=0.01, early_stopping=False)
    test_loss, accuracy = n1_mse_no_early_stop.evaluate(X_test, y_test)
    print(f"N1 with MSE (without early stopping) - Test Loss: {test_loss}, Test Accuracy: {accuracy}")

    print("Training N1 with MSE (with early stopping)...")
    n1_mse_early_stop = NeuralNetwork(n1_layers, loss_function='mse')
    n1_mse_early_stop.train(X_train, y_train, X_test, y_test, epochs=100, batch_size=40, learning_rate=0.01, early_stopping=True, patience=10)
    test_loss, accuracy = n1_mse_early_stop.evaluate(X_test, y_test)
    print(f"N1 with MSE (with early stopping) - Test Loss: {test_loss}, Test Accuracy: {accuracy}")

    plt.figure(figsize=(12, 6))
    plt.plot(n1_mse_no_early_stop.train_loss_history, label='Training Loss (No Early Stopping)')
    plt.plot(n1_mse_no_early_stop.val_loss_history, label='Validation Loss (No Early Stopping)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('N1 with MSE - Loss History (Without Early Stopping)')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(n1_mse_early_stop.train_loss_history, label='Training Loss (With Early Stopping)')
    plt.plot(n1_mse_early_stop.val_loss_history, label='Validation Loss (With Early Stopping)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('N1 with MSE - Loss History (With Early Stopping)')
    plt.legend()
    plt.show() 