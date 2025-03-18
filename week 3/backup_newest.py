import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

class NeuralNetwork:
    def __init__(self, optimizer='SGD', lr=0.01, beta=0.9, beta2=0.999, epsilon=1e-8, epochs=1000):
        self.optimizer = optimizer
        self.lr = lr
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        self.epochs = epochs
        self.w = np.random.randn()
        self.b = np.random.randn()
        self.history = []
        self.loss_history = []
        self.vdw, self.sdw = 0, 0
        self.vdb, self.sdb = 0, 0
        self.t = 0

    def forward(self, X):
        return self.w * X + self.b

    def compute_loss(self, Y_pred, Y):
        m = len(Y)
        return (1/(2*m)) * np.sum((Y_pred - Y) ** 2)

    def backward(self, X, Y):
        m = len(Y)
        Y_pred = self.forward(X)
        dw = (1/m) * np.sum((Y_pred - Y) * X)
        db = (1/m) * np.sum(Y_pred - Y)
        loss = self.compute_loss(Y_pred, Y)
        self.history.append((self.w, self.b, loss))
        self.loss_history.append(loss)
        self.update_parameters(dw, db)

    def update_parameters(self, dw, db):
        if self.optimizer == 'SGD':
            self.w -= self.lr * dw
            self.b -= self.lr * db
        elif self.optimizer == 'Momentum':
            self.vdw = self.beta * self.vdw + (1 - self.beta) * dw
            self.vdb = self.beta * self.vdb + (1 - self.beta) * db
            self.w -= self.lr * self.vdw
            self.b -= self.lr * self.vdb
        elif self.optimizer == 'RMSprop':
            self.sdw = self.beta * self.sdw + (1 - self.beta) * dw**2
            self.sdb = self.beta * self.sdb + (1 - self.beta) * db**2
            self.w -= self.lr * dw / (np.sqrt(self.sdw + self.epsilon))
            self.b -= self.lr * db / (np.sqrt(self.sdb + self.epsilon))
        elif self.optimizer == 'Adam':
            self.t += 1
            self.vdw = self.beta * self.vdw + (1 - self.beta) * dw
            self.vdb = self.beta * self.vdb + (1 - self.beta) * db
            self.sdw = self.beta2 * self.sdw + (1 - self.beta2) * dw**2
            self.sdb = self.beta2 * self.sdb + (1 - self.beta2) * db**2
            vdw_corrected = self.vdw / (1 - self.beta**self.t)
            vdb_corrected = self.vdb / (1 - self.beta**self.t)
            sdw_corrected = self.sdw / (1 - self.beta2**self.t)
            sdb_corrected = self.sdb / (1 - self.beta2**self.t)
            self.w -= self.lr * vdw_corrected / (np.sqrt(sdw_corrected) + self.epsilon)
            self.b -= self.lr * vdb_corrected / (np.sqrt(sdb_corrected) + self.epsilon)
        elif self.optimizer == 'Adagrad':
            self.sdw += dw**2
            self.sdb += db**2
            self.w -= self.lr * dw / (np.sqrt(self.sdw + self.epsilon))
            self.b -= self.lr * db / (np.sqrt(self.sdb + self.epsilon))

    def train(self, X, Y):
        progress_bar = tqdm(range(self.epochs), desc=f'Training ({self.optimizer})')
        for epoch in progress_bar:
            self.backward(X, Y)
            if epoch % 100 == 0:
                progress_bar.set_postfix(loss=f"{self.loss_history[-1]:.4f}")

np.random.seed(42)
X = np.array([np.linspace(-10, 10, 100), np.linspace(-10, 10, 100)])
X = X.reshape(100, 2)
print(X.shape)
Y = 3 * X + 7 + np.sin(X) * 5 + np.random.randn(*X.shape) * 4

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def compute_loss_surface(model, X, Y, w_range=(-5, 5), b_range=(-5, 5), resolution=100):
    w_vals = np.linspace(w_range[0], w_range[1], resolution)
    b_vals = np.linspace(b_range[0], b_range[1], resolution)
    W, B = np.meshgrid(w_vals, b_vals)
    loss_surface = np.zeros_like(W)
    for i in range(resolution):
        for j in range(resolution):
            model.w = W[i, j]
            model.b = B[i, j]
            loss_surface[i, j] = model.compute_loss(model.forward(X), Y)
    return W, B, loss_surface

def animate_training(model, W, B, loss_surface):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(W, B, loss_surface, cmap='viridis', alpha=0.7)
    path = np.array(model.history)
    line, = ax.plot([], [], [], 'r.-', label='Optimization Path')
    
    def update(frame):
        if frame < len(path):
            line.set_data(path[:frame+1, 0], path[:frame+1, 1])
            line.set_3d_properties(path[:frame+1, 2])
        return line,
    
    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=50, blit=False)
    ax.set_xlabel('Weight')
    ax.set_ylabel('Bias')
    ax.set_zlabel('Loss')
    ax.set_title(f'Loss Surface and Optimization Path ({model.optimizer})')
    ax.legend()
    plt.show()

# optimizers = ['SGD', 'Momentum', 'RMSprop', 'Adam', 'Adagrad']
# for optimizer in optimizers:
#     model = NeuralNetwork(optimizer=optimizer, epochs=1000, lr=0.01 if optimizer in ['SGD', 'Momentum'] else 0.001)
#     model.train(X_train, Y_train)
#     W, B, loss_surface = compute_loss_surface(model, X_train, Y_train)
#     animate_training(model, W, B, loss_surface)


class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Hyperparameter Tuning")

        ttk.Label(root, text="Learning Rate:").grid(row=0, column=0, padx=10, pady=10)
        self.lr_entry = ttk.Entry(root)
        self.lr_entry.grid(row=0, column=1, padx=10, pady=10)
        self.lr_entry.insert(0, "0.01")

        ttk.Label(root, text="Epochs:").grid(row=1, column=0, padx=10, pady=10)
        self.epochs_entry = ttk.Entry(root)
        self.epochs_entry.grid(row=1, column=1, padx=10, pady=10)
        self.epochs_entry.insert(0, "1000")

        ttk.Label(root, text="Optimizer:").grid(row=2, column=0, padx=10, pady=10)
        self.optimizer_var = tk.StringVar()
        self.optimizer_combobox = ttk.Combobox(root, textvariable=self.optimizer_var)
        self.optimizer_combobox['values'] = ('SGD', 'Momentum', 'RMSprop', 'Adam', 'Adagrad')
        self.optimizer_combobox.grid(row=2, column=1, padx=10, pady=10)
        self.optimizer_combobox.current(0)

        self.train_button = ttk.Button(root, text="Train", command=self.start_training)
        self.train_button.grid(row=3, column=0, columnspan=2, pady=10)

    def start_training(self):
        lr = float(self.lr_entry.get())
        epochs = int(self.epochs_entry.get())
        optimizer = self.optimizer_var.get()

        model = NeuralNetwork(optimizer=optimizer, epochs=epochs, lr=lr)
        model.train(X_train, Y_train)

        W, B, loss_surface = compute_loss_surface(model, X_train, Y_train)
        animate_training(model, W, B, loss_surface)

root = tk.Tk()
app = NeuralNetworkApp(root)
root.mainloop()