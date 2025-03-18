import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import ttk

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    encoder = LabelEncoder()
    df['ObesityCategory'] = encoder.fit_transform(df['ObesityCategory'])
    
    X = df[['Age', 'Gender', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']].values
    y = df['ObesityCategory'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42), encoder

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2, use_dropout=True):
        self.layers = []
        self.biases = []
        self.use_dropout = use_dropout
        self.training = True  # Training mode flag
        
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            # Xavier initialization
            self.layers.append(np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2 / sizes[i]))
            self.biases.append(np.zeros((1, sizes[i+1])))

    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.a = [X]
        for i in range(len(self.layers) - 1):
            X = np.dot(X, self.layers[i]) + self.biases[i]
            X = self.relu(X)
            if self.training and self.use_dropout:
                mask = (np.random.rand(*X.shape) > 0.2) / 0.8
                X *= mask
            self.a.append(X)
        output = self.softmax(np.dot(X, self.layers[-1]) + self.biases[-1])
        self.a.append(output)
        return output

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(np.log(y_pred[np.arange(m), y_true] + 1e-8)) / m

    def compute_accuracy(self, y_true, y_pred):
        y_pred_labels = np.argmax(y_pred, axis=1)
        return np.sum(y_pred_labels == y_true) / len(y_true) * 100

    def backward(self, y):
        m = y.shape[0]
        y_pred = self.a[-1]
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(m), y] = 1
        dZ = y_pred - y_one_hot

        for i in range(len(self.layers) - 1, -1, -1):
            dW = np.dot(self.a[i].T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            dZ = np.dot(dZ, self.layers[i].T) * self.relu_derivative(self.a[i])
            self.layers[i] -= self.lr * dW
            self.biases[i] -= self.lr * dB

    def train(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=40, learning_rate=0.01):
        self.lr = learning_rate
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]

                self.forward(X_batch)
                self.backward(y_batch)

            # Learning rate decay
            if epoch % 100 == 0 and epoch != 0:
                self.lr *= 0.95

            self.training = False  # Disable dropout during validation
            y_train_pred = self.forward(X_train)
            y_val_pred = self.forward(X_val)
            self.training = True  # Re-enable dropout

            train_loss = self.compute_loss(y_train, y_train_pred)
            val_loss = self.compute_loss(y_val, y_val_pred)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        print("Final Accuracy:")
        print("Train:", self.compute_accuracy(y_train, self.forward(X_train)))
        print("Validation:", self.compute_accuracy(y_val, self.forward(X_val)))

        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.legend()
        plt.show()

(filepath,) = ("./obesity_data.csv",)
(X_train, X_test, y_train, y_test), encoder = load_data(filepath)

def train_nn():
    global encoder  
    hidden_layers = int(hidden_layers_var.get())
    neurons_per_layer = int(neurons_var.get())
    learning_rate = float(lr_var.get())
    batch_size = int(batch_size_var.get())
    epochs = int(epochs_var.get())
    loss_function = loss_var.get()
    early = early_var.get()
    use_dropout = dropout_var.get()  # Get dropout selection

    hidden_sizes = [neurons_per_layer] * hidden_layers
    nn = NeuralNetwork(input_size=6, hidden_sizes=hidden_sizes, output_size=len(encoder.classes_), use_dropout=use_dropout)
    nn.train(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate, loss_function )

root = tk.Tk()
root.title("Neural Network Trainer")

tk.Label(root, text="Hidden Layers:").grid(row=0, column=0)
hidden_layers_var = tk.StringVar(value="3")
ttk.Entry(root, textvariable=hidden_layers_var).grid(row=0, column=1)

tk.Label(root, text="Neurons per Layer:").grid(row=1, column=0)
neurons_var = tk.StringVar(value="16")
ttk.Entry(root, textvariable=neurons_var).grid(row=1, column=1)

tk.Label(root, text="Learning Rate:").grid(row=2, column=0)
lr_var = tk.StringVar(value="0.01")
ttk.Entry(root, textvariable=lr_var).grid(row=2, column=1)

tk.Label(root, text="Batch Size:").grid(row=3, column=0)
batch_size_var = tk.StringVar(value="40")
ttk.Entry(root, textvariable=batch_size_var).grid(row=3, column=1)

tk.Label(root, text="Epochs:").grid(row=4, column=0)
epochs_var = tk.StringVar(value="1000")
ttk.Entry(root, textvariable=epochs_var).grid(row=4, column=1)

tk.Label(root, text="Loss Function:").grid(row=5, column=0)
loss_var = tk.StringVar(value="cross_entropy")
ttk.Combobox(root, textvariable=loss_var, values=["cross_entropy", "mse"]).grid(row=5, column=1)

tk.Label(root, text="Early Stopping").grid(row=6, column=0)
early_var = tk.BooleanVar(value=False)
ttk.Checkbutton(root, variable=early_var).grid(row=6, column=1)

tk.Label(root, text="Use Dropout").grid(row=7, column=0)
dropout_var = tk.BooleanVar(value=True)
ttk.Checkbutton(root, variable=dropout_var).grid(row=7, column=1)

ttk.Button(root, text="Train", command=train_nn).grid(row=8, columnspan=2)
root.mainloop()