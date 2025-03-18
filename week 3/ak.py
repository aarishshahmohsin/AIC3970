import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# Generate a 3D synthetic dataset
np.random.seed(42)
X = np.random.uniform(-5, 5, (3, 500))  # 3D input
y = np.sin(X[0]) + np.cos(X[1]) + 0.5 * X[2] + np.random.normal(0, 0.2, 500)  # Target

# Normalize data
X_mean, X_std = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
X = (X - X_mean) / X_std
y = (y - y.mean()) / y.std()

class NeuralNetwork:
    def __init__(self, input_size=3, hidden_size=5, output_size=1, optimizer='SGD', lr=0.01, epochs=1000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))
        self.history = []

    def activation(self, Z):
        return np.maximum(0, Z)

    def activation_derivative(self, Z):
        return (Z > 0).astype(float)

    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.W2, self.A1) + self.b2
        return self.Z2

    def compute_loss(self, Y_pred, Y):
        return np.mean((Y_pred - Y) ** 2)

    def backward(self, X, Y):
        m = X.shape[1]
        dZ2 = self.Z2 - Y
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.history.append((self.W1[0, 0], self.W1[1, 0], self.compute_loss(self.forward(X), Y)))

    def train(self, X, Y):
        X = X.T
        Y = Y.reshape(1, -1)
        for _ in tqdm(range(self.epochs), desc="Training"):
            Y_pred = self.forward(X)
            self.backward(X, Y)

def compute_loss_surface(nn, X, Y, w1_range=(-1, 1), w2_range=(-1, 1), resolution=50):
    w1_vals = np.linspace(w1_range[0], w1_range[1], resolution)
    w2_vals = np.linspace(w2_range[0], w2_range[1], resolution)
    loss_surface = np.zeros((resolution, resolution))
    original_W1 = nn.W1.copy()

    for i, w1 in enumerate(w1_vals):
        for j, w2 in enumerate(w2_vals):
            nn.W1[0, 0] = w1
            nn.W1[1, 0] = w2
            Y_pred = nn.forward(X.T)
            loss_surface[i, j] = nn.compute_loss(Y_pred, Y.reshape(1, -1))

    nn.W1 = original_W1.copy()
    return w1_vals, w2_vals, loss_surface

def plot_loss_surface(w1_vals, w2_vals, loss_surface, optimizer_path=None):
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W1, W2, loss_surface, cmap='viridis', alpha=0.8)
    
    if optimizer_path is not None:
        path = np.array(optimizer_path)
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color='r', marker='o', markersize=3, label="Optimizer Trajectory")

    ax.set_xlabel('Weight W1[0,0]')
    ax.set_ylabel('Weight W1[1,0]')
    ax.set_zlabel('Loss')
    ax.set_title('3D Loss Surface with Optimizer Path')
    ax.legend()
    plt.show()

# Train network and plot results
nn = NeuralNetwork(optimizer='SGD', epochs=200)
nn.train(X, y)

w1_vals, w2_vals, loss_surface = compute_loss_surface(nn, X, y)
plot_loss_surface(w1_vals, w2_vals, loss_surface, optimizer_path=nn.history)
