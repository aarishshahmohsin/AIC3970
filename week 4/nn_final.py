import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm

class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, problem_type='regression'):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.problem_type = problem_type

        self.W1 = np.random.randn(input_size, hidden1_size) * 0.01
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.01
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))

        self.W1_history = [self.W1.copy()]
        self.b1_history = [self.b1.copy()]
        self.W2_history = [self.W2.copy()]
        self.b2_history = [self.b2.copy()]
        self.W3_history = [self.W3.copy()]
        self.b3_history = [self.b3.copy()]
        self.loss_history = []

    def reset(self):
        self.W1 = np.random.randn(self.input_size, self.hidden1_size) * 0.01
        self.b1 = np.zeros((1, self.hidden1_size))
        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * 0.01
        self.b2 = np.zeros((1, self.hidden2_size))
        self.W3 = np.random.randn(self.hidden2_size, self.output_size) * 0.01
        self.b3 = np.zeros((1, self.output_size))

        self.W1_history = [self.W1.copy()]
        self.b1_history = [self.b1.copy()]
        self.W2_history = [self.W2.copy()]
        self.b2_history = [self.b2.copy()]
        self.W3_history = [self.W3.copy()]
        self.b3_history = [self.b3.copy()]
        self.loss_history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3

        if self.problem_type == 'regression':
            self.output = self.z3  # Linear activation for regression
        else:  # classification
            if self.output_size > 1:
                self.output = self.softmax(self.z3)  # Softmax for multi-class
            else:
                self.output = self.sigmoid(self.z3)  # Sigmoid for binary

        return self.output

    def compute_loss(self, y_pred, y_true):
        if self.problem_type == 'regression':
            loss = np.mean(np.square(y_pred - y_true))
        else:  # classification
            if self.output_size > 1:
                loss = -np.mean(y_true * np.log(y_pred + 1e-8))
            else:
                loss = -np.mean(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8))

        return loss

    def backward(self, X, y):
        batch_size = X.shape[0]

        if self.problem_type == 'regression':
            dz3 = (self.output - y) / batch_size
        else:  # classification
            if self.output_size > 1:
                dz3 = (self.output - y) / batch_size
            else:
                dz3 = (self.output - y) / batch_size

        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)

        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * self.relu_derivative(self.a2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.a1)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2, dW3, db3

    def train_sgd(self, X, y, learning_rate=0.01, epochs=100, batch_size=32):
        self.reset()
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)

                loss = self.compute_loss(y_pred, y_batch)
                self.loss_history.append(loss)

                dW1, db1, dW2, db2, dW3, db3 = self.backward(X_batch, y_batch)

                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W3 -= learning_rate * dW3
                self.b3 -= learning_rate * db3

                self.W1_history.append(self.W1.copy())
                self.b1_history.append(self.b1.copy())
                self.W2_history.append(self.W2.copy())
                self.b2_history.append(self.b2.copy())
                self.W3_history.append(self.W3.copy())
                self.b3_history.append(self.b3.copy())

        return self.loss_history

    def train_momentum(self, X, y, learning_rate=0.01, beta=0.9, epochs=100, batch_size=32):
        self.reset()
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        vW1 = np.zeros_like(self.W1)
        vb1 = np.zeros_like(self.b1)
        vW2 = np.zeros_like(self.W2)
        vb2 = np.zeros_like(self.b2)
        vW3 = np.zeros_like(self.W3)
        vb3 = np.zeros_like(self.b3)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)

                loss = self.compute_loss(y_pred, y_batch)
                self.loss_history.append(loss)

                dW1, db1, dW2, db2, dW3, db3 = self.backward(X_batch, y_batch)

                vW1 = beta * vW1 + (1 - beta) * dW1
                vb1 = beta * vb1 + (1 - beta) * db1
                vW2 = beta * vW2 + (1 - beta) * dW2
                vb2 = beta * vb2 + (1 - beta) * db2
                vW3 = beta * vW3 + (1 - beta) * dW3
                vb3 = beta * vb3 + (1 - beta) * db3

                self.W1 -= learning_rate * vW1
                self.b1 -= learning_rate * vb1
                self.W2 -= learning_rate * vW2
                self.b2 -= learning_rate * vb2
                self.W3 -= learning_rate * vW3
                self.b3 -= learning_rate * vb3

                self.W1_history.append(self.W1.copy())
                self.b1_history.append(self.b1.copy())
                self.W2_history.append(self.W2.copy())
                self.b2_history.append(self.b2.copy())
                self.W3_history.append(self.W3.copy())
                self.b3_history.append(self.b3.copy())

        return self.loss_history

    def train_rmsprop(self, X, y, learning_rate=0.01, beta=0.9, epsilon=1e-8, epochs=100, batch_size=32):
        self.reset()
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        sW1 = np.zeros_like(self.W1)
        sb1 = np.zeros_like(self.b1)
        sW2 = np.zeros_like(self.W2)
        sb2 = np.zeros_like(self.b2)
        sW3 = np.zeros_like(self.W3)
        sb3 = np.zeros_like(self.b3)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)

                loss = self.compute_loss(y_pred, y_batch)
                self.loss_history.append(loss)

                dW1, db1, dW2, db2, dW3, db3 = self.backward(X_batch, y_batch)

                sW1 = beta * sW1 + (1 - beta) * np.square(dW1)
                sb1 = beta * sb1 + (1 - beta) * np.square(db1)
                sW2 = beta * sW2 + (1 - beta) * np.square(dW2)
                sb2 = beta * sb2 + (1 - beta) * np.square(db2)
                sW3 = beta * sW3 + (1 - beta) * np.square(dW3)
                sb3 = beta * sb3 + (1 - beta) * np.square(db3)

                self.W1 -= learning_rate * dW1 / (np.sqrt(sW1) + epsilon)
                self.b1 -= learning_rate * db1 / (np.sqrt(sb1) + epsilon)
                self.W2 -= learning_rate * dW2 / (np.sqrt(sW2) + epsilon)
                self.b2 -= learning_rate * db2 / (np.sqrt(sb2) + epsilon)
                self.W3 -= learning_rate * dW3 / (np.sqrt(sW3) + epsilon)
                self.b3 -= learning_rate * db3 / (np.sqrt(sb3) + epsilon)

                self.W1_history.append(self.W1.copy())
                self.b1_history.append(self.b1.copy())
                self.W2_history.append(self.W2.copy())
                self.b2_history.append(self.b2.copy())
                self.W3_history.append(self.W3.copy())
                self.b3_history.append(self.b3.copy())

        return self.loss_history

    def train_adam(self, X, y, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, epochs=100, batch_size=32):
        self.reset()
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        mW3, vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        mb3, vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)

        t = 0

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                t += 1
                start = i * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)

                loss = self.compute_loss(y_pred, y_batch)
                self.loss_history.append(loss)

                dW1, db1, dW2, db2, dW3, db3 = self.backward(X_batch, y_batch)

                mW1 = beta1 * mW1 + (1 - beta1) * dW1
                mb1 = beta1 * mb1 + (1 - beta1) * db1
                mW2 = beta1 * mW2 + (1 - beta1) * dW2
                mb2 = beta1 * mb2 + (1 - beta1) * db2
                mW3 = beta1 * mW3 + (1 - beta1) * dW3
                mb3 = beta1 * mb3 + (1 - beta1) * db3

                vW1 = beta2 * vW1 + (1 - beta2) * np.square(dW1)
                vb1 = beta2 * vb1 + (1 - beta2) * np.square(db1)
                vW2 = beta2 * vW2 + (1 - beta2) * np.square(dW2)
                vb2 = beta2 * vb2 + (1 - beta2) * np.square(db2)
                vW3 = beta2 * vW3 + (1 - beta2) * np.square(dW3)
                vb3 = beta2 * vb3 + (1 - beta2) * np.square(db3)

                mW1_corrected = mW1 / (1 - beta1 ** t)
                mb1_corrected = mb1 / (1 - beta1 ** t)
                mW2_corrected = mW2 / (1 - beta1 ** t)
                mb2_corrected = mb2 / (1 - beta1 ** t)
                mW3_corrected = mW3 / (1 - beta1 ** t)
                mb3_corrected = mb3 / (1 - beta1 ** t)

                vW1_corrected = vW1 / (1 - beta2 ** t)
                vb1_corrected = vb1 / (1 - beta2 ** t)
                vW2_corrected = vW2 / (1 - beta2 ** t)
                vb2_corrected = vb2 / (1 - beta2 ** t)
                vW3_corrected = vW3 / (1 - beta2 ** t)
                vb3_corrected = vb3 / (1 - beta2 ** t)

                self.W1 -= learning_rate * mW1_corrected / (np.sqrt(vW1_corrected) + epsilon)
                self.b1 -= learning_rate * mb1_corrected / (np.sqrt(vb1_corrected) + epsilon)
                self.W2 -= learning_rate * mW2_corrected / (np.sqrt(vW2_corrected) + epsilon)
                self.b2 -= learning_rate * mb2_corrected / (np.sqrt(vb2_corrected) + epsilon)
                self.W3 -= learning_rate * mW3_corrected / (np.sqrt(vW3_corrected) + epsilon)
                self.b3 -= learning_rate * mb3_corrected / (np.sqrt(vb3_corrected) + epsilon)

                self.W1_history.append(self.W1.copy())
                self.b1_history.append(self.b1.copy())
                self.W2_history.append(self.W2.copy())
                self.b2_history.append(self.b2.copy())
                self.W3_history.append(self.W3.copy())
                self.b3_history.append(self.b3.copy())

        return self.loss_history

    def train_adagrad(self, X, y, learning_rate=0.01, epsilon=1e-8, epochs=100, batch_size=32):
        self.reset()
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / batch_size))

        gW1 = np.zeros_like(self.W1)
        gb1 = np.zeros_like(self.b1)
        gW2 = np.zeros_like(self.W2)
        gb2 = np.zeros_like(self.b2)
        gW3 = np.zeros_like(self.W3)
        gb3 = np.zeros_like(self.b3)

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_samples)

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Compute loss
                loss = self.compute_loss(y_pred, y_batch)
                self.loss_history.append(loss)

                dW1, db1, dW2, db2, dW3, db3 = self.backward(X_batch, y_batch)

                gW1 += np.square(dW1)
                gb1 += np.square(db1)
                gW2 += np.square(dW2)
                gb2 += np.square(db2)
                gW3 += np.square(dW3)
                gb3 += np.square(db3)

                self.W1 -= learning_rate * dW1 / (np.sqrt(gW1) + epsilon)
                self.b1 -= learning_rate * db1 / (np.sqrt(gb1) + epsilon)
                self.W2 -= learning_rate * dW2 / (np.sqrt(gW2) + epsilon)
                self.b2 -= learning_rate * db2 / (np.sqrt(gb2) + epsilon)
                self.W3 -= learning_rate * dW3 / (np.sqrt(gW3) + epsilon)
                self.b3 -= learning_rate * db3 / (np.sqrt(gb3) + epsilon)

                self.W1_history.append(self.W1.copy())
                self.b1_history.append(self.b1.copy())
                self.W2_history.append(self.W2.copy())
                self.b2_history.append(self.b2.copy())
                self.W3_history.append(self.W3.copy())
                self.b3_history.append(self.b3.copy())

        return self.loss_history

def generate_regression_data(n_samples=100, noise=0.1):
    np.random.seed(0)
    X = np.random.rand(n_samples, 1) * 10 - 5  # Random values between -5 and 5
    y = 0.5 * X**2 + 2 * X + 1 + noise * np.random.randn(n_samples, 1)
    return X, y

def generate_classification_data(n_samples=100, n_features=2, n_classes=2):
    np.random.seed(0)
    if n_classes == 2:
        X = np.random.randn(n_samples, n_features)
        y_temp = np.sum(X**2, axis=1)
        y = np.zeros((n_samples, 1))
        y[y_temp > 2] = 1
    else:
        X = np.random.randn(n_samples, n_features)
        y_temp = np.sum(X**2, axis=1)
        y = np.zeros((n_samples, n_classes))
        thresholds = np.linspace(0, 10, n_classes+1)
        for i in range(n_classes):
            mask = (y_temp >= thresholds[i]) & (y_temp < thresholds[i+1])
            y[mask, i] = 1

    return X, y

class NeuralNetworkApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Neural Network Optimization Visualizer")
        self.master.geometry("1200x800")

        self.control_frame = ttk.Frame(master, padding="10")
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(self.control_frame, text="Problem Type:").grid(row=0, column=0, padx=5, pady=5)
        self.problem_type_var = tk.StringVar(value="regression")
        ttk.Radiobutton(self.control_frame, text="Regression", variable=self.problem_type_var, 
                        value="regression").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(self.control_frame, text="Classification", variable=self.problem_type_var, 
                        value="classification").grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Network Architecture:").grid(row=1, column=0, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Input Size:").grid(row=1, column=1, padx=5, pady=5)
        self.input_size_var = tk.IntVar(value=1)
        ttk.Entry(self.control_frame, textvariable=self.input_size_var, width=5).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Hidden1 Size:").grid(row=1, column=3, padx=5, pady=5)
        self.hidden1_size_var = tk.IntVar(value=2)
        ttk.Entry(self.control_frame, textvariable=self.hidden1_size_var, width=5).grid(row=1, column=4, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Hidden2 Size:").grid(row=1, column=5, padx=5, pady=5)
        self.hidden2_size_var = tk.IntVar(value=3)
        ttk.Entry(self.control_frame, textvariable=self.hidden2_size_var, width=5).grid(row=1, column=6, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Output Size:").grid(row=1, column=7, padx=5, pady=5)
        self.output_size_var = tk.IntVar(value=1)
        ttk.Entry(self.control_frame, textvariable=self.output_size_var, width=5).grid(row=1, column=8, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Training Parameters:").grid(row=2, column=0, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Learning Rate:").grid(row=2, column=1, padx=5, pady=5)
        self.learning_rate_var = tk.DoubleVar(value=0.01)
        ttk.Entry(self.control_frame, textvariable=self.learning_rate_var, width=5).grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Epochs:").grid(row=2, column=3, padx=5, pady=5)
        self.epochs_var = tk.IntVar(value=50)
        ttk.Entry(self.control_frame, textvariable=self.epochs_var, width=5).grid(row=2, column=4, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Batch Size:").grid(row=2, column=5, padx=5, pady=5)
        self.batch_size_var = tk.IntVar(value=10)
        ttk.Entry(self.control_frame, textvariable=self.batch_size_var, width=5).grid(row=2, column=6, padx=5, pady=5)

        ttk.Label(self.control_frame, text="Optimizer:").grid(row=3, column=0, padx=5, pady=5)
        self.optimizer_var = tk.StringVar(value="sgd")
        ttk.Radiobutton(self.control_frame, text="SGD", variable=self.optimizer_var, 
                        value="sgd").grid(row=3, column=1, padx=5, pady=5)
        ttk.Radiobutton(self.control_frame, text="Momentum", variable=self.optimizer_var, 
                        value="momentum").grid(row=3, column=2, padx=5, pady=5)
        ttk.Radiobutton(self.control_frame, text="RMSprop", variable=self.optimizer_var, 
                        value="rmsprop").grid(row=3, column=3, padx=5, pady=5)
        ttk.Radiobutton(self.control_frame, text="Adam", variable=self.optimizer_var, 
                        value="adam").grid(row=3, column=4, padx=5, pady=5)
        ttk.Radiobutton(self.control_frame, text="Adagrad", variable=self.optimizer_var, 
                        value="adagrad").grid(row=3, column=5, padx=5, pady=5)

        self.train_button = ttk.Button(self.control_frame, text="Train", command=self.train)
        self.train_button.grid(row=4, column=0, padx=5, pady=10)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.control_frame, textvariable=self.status_var).grid(row=4, column=1, columnspan=6, padx=5, pady=5, sticky=tk.W)

        self.tab_control = ttk.Notebook(master)
        self.tab_control.pack(fill=tk.BOTH, expand=1)

        self.loss_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.loss_tab, text="Loss Curve")

        self.params_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.params_tab, text="Weight & Bias Values")

        self.vis_tab = ttk.Frame(self.tab_control)
        self.tab_control.add(self.vis_tab, text="3D Visualization")

        self.fig_loss = Figure(figsize=(10, 6))
        self.ax_loss = self.fig_loss.add_subplot(111)
        self.canvas_loss = FigureCanvasTkAgg(self.fig_loss, master=self.loss_tab)
        self.canvas_loss.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        self.params_frame = ttk.Frame(self.params_tab)
        self.params_frame.pack(fill=tk.BOTH, expand=1)

        self.weight_selection_frame = ttk.Frame(self.vis_tab)
        self.weight_selection_frame.pack(fill=tk.X)

        ttk.Label(self.weight_selection_frame, text="Select Weight:").pack(side=tk.LEFT, padx=5, pady=5)
        self.weight_var = tk.StringVar(value="W1[0,0]")
        self.weight_combo = ttk.Combobox(self.weight_selection_frame, textvariable=self.weight_var, state="readonly")
        self.weight_combo.pack(side=tk.LEFT, padx=5, pady=5)

        self.fig_vis = Figure(figsize=(10, 6))
        self.ax_vis = self.fig_vis.add_subplot(111, projection='3d')
        self.canvas_vis = FigureCanvasTkAgg(self.fig_vis, master=self.vis_tab)
        self.canvas_vis.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        self.model = None
        self.data_generated = False

    def generate_data(self):
        problem_type = self.problem_type_var.get()
        input_size = self.input_size_var.get()
        output_size = self.output_size_var.get()

        if problem_type == "regression":
            X, y = generate_regression_data(n_samples=100)
        else:  # classification
            X, y = generate_classification_data(n_samples=100, n_features=input_size, n_classes=output_size)

        self.X = X
        self.y = y
        self.data_generated = True

    def init_model(self):
        input_size = self.input_size_var.get()
        hidden1_size = self.hidden1_size_var.get()
        hidden2_size = self.hidden2_size_var.get()
        output_size = self.output_size_var.get()
        problem_type = self.problem_type_var.get()

        self.model = NeuralNetwork(input_size, hidden1_size, hidden2_size, output_size, problem_type)

    def train(self):
        if not self.data_generated:
            self.generate_data()

        if self.model is None:
            self.init_model()

        learning_rate = self.learning_rate_var.get()
        epochs = self.epochs_var.get()
        batch_size = self.batch_size_var.get()
        optimizer = self.optimizer_var.get()

        self.status_var.set(f"Training with {optimizer}...")
        self.master.update()

        if optimizer == "sgd":
            self.model.train_sgd(self.X, self.y, learning_rate, epochs, batch_size)
        elif optimizer == "momentum":
            self.model.train_momentum(self.X, self.y, learning_rate, 0.9, epochs, batch_size)
        elif optimizer == "rmsprop":
            self.model.train_rmsprop(self.X, self.y, learning_rate, 0.9, 1e-8, epochs, batch_size)
        elif optimizer == "adam":
            self.model.train_adam(self.X, self.y, learning_rate, 0.9, 0.999, 1e-8, epochs, batch_size)
        elif optimizer == "adagrad":
            self.model.train_adagrad(self.X, self.y, learning_rate, 1e-8, epochs, batch_size)

        self.status_var.set(f"Training complete with {optimizer}")

        self.display_loss_curve()
        self.display_parameter_table()
        self.setup_weight_selection()
        self.display_3d_visualization()

    def display_loss_curve(self):
        self.ax_loss.clear()

        self.ax_loss.plot(self.model.loss_history)
        self.ax_loss.set_title(f'Loss Curve - {self.optimizer_var.get().upper()}')
        self.ax_loss.set_xlabel('Iterations')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True)

        self.canvas_loss.draw()

    def display_parameter_table(self):
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.params_frame)
        scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        headers = ["Iteration", "W1[0,0]", "b1[0,0]", "W2[0,0]", "b2[0,0]", "W3[0,0]", "b3[0,0]", "Loss"]
        for i, header in enumerate(headers):
            ttk.Label(scrollable_frame, text=header, font=('TkDefaultFont', 10, 'bold')).grid(row=0, column=i, padx=5, pady=5)

        for i in range(min(len(self.model.loss_history), 100)):  # Limit to 100 rows for performance
            ttk.Label(scrollable_frame, text=str(i)).grid(row=i+1, column=0, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=f"{self.model.W1_history[i][0, 0]:.6f}").grid(row=i+1, column=1, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=f"{self.model.b1_history[i][0, 0]:.6f}").grid(row=i+1, column=2, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=f"{self.model.W2_history[i][0, 0]:.6f}").grid(row=i+1, column=3, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=f"{self.model.b2_history[i][0, 0]:.6f}").grid(row=i+1, column=4, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=f"{self.model.W3_history[i][0, 0]:.6f}").grid(row=i+1, column=5, padx=5, pady=2)
            ttk.Label(scrollable_frame, text=f"{self.model.b3_history[i][0, 0]:.6f}").grid(row=i+1, column=6, padx=5, pady=2)
            if i < len(self.model.loss_history):
                ttk.Label(scrollable_frame, text=f"{self.model.loss_history[i]:.6f}").grid(row=i+1, column=7, padx=5, pady=2)

    def setup_weight_selection(self):
        options = []
        for i in range(min(3, self.model.W1.shape[0])):
            for j in range(min(3, self.model.W1.shape[1])):
                options.append(f"W1[{i},{j}]")

        for i in range(min(3, self.model.W2.shape[0])):
            for j in range(min(3, self.model.W2.shape[1])):
                options.append(f"W2[{i},{j}]")

        for i in range(min(3, self.model.W3.shape[0])):
            for j in range(min(3, self.model.W3.shape[1])):
                options.append(f"W3[{i},{j}]")

        self.weight_combo['values'] = options
        self.weight_combo.bind("<<ComboboxSelected>>", self.display_3d_visualization)

    def display_3d_visualization(self, event=None):
        if self.model is None or not self.data_generated:
            return

        self.ax_vis.clear()

        weight_str = self.weight_var.get()
        weight_matrix, indices = self.parse_weight_string(weight_str)

        iterations = len(self.model.loss_history)
        weight_values = np.array([getattr(self.model, f"{weight_matrix}_history")[i][indices] for i in range(iterations)])

        iterations_array = np.arange(iterations)

        self.ax_vis.plot(iterations_array, weight_values, self.model.loss_history, 'r-', linewidth=2)

        self.ax_vis.set_xlabel('Iterations')
        self.ax_vis.set_ylabel(f'{weight_str} Value')
        self.ax_vis.set_zlabel('Loss')
        self.ax_vis.set_title(f'Optimization Path - {self.optimizer_var.get().upper()}')

        self.canvas_vis.draw()

    def parse_weight_string(self, weight_str):
        matrix_name = weight_str.split('[')[0]
        indices_str = weight_str.split('[')[1].strip(']')
        i, j = map(int, indices_str.split(','))
        return matrix_name, (i, j)

# X, y = generate_regression_data()
# plt.plot(X, y)
# plt.show()

# X, y = generate_classification_data()
# # print(X[:, 0].shape, y.shape)
# # print(X[:, 0], y)
# print(X.shape, y.shape)
# # print(X[:, 0].shape, X[:, 1].shape)
# y = y.reshape(-1)
# true = X[y == 1]
# false = X[y == 0]
# plt.scatter(true[:,0], true[:,1], color='red')
# plt.scatter(false[:,0], false[:,1], color='blue')
# plt.show()

# plt.close('all')
# X, y = generate_regression_data()
# print(X.shape, y.shape)
# plt.scatter(X.T, y)

def main():
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
