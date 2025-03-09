import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tkinter as tk 
from tkinter import ttk

# Load data from CSV
def load_data(filepath):
    df = pd.read_csv(filepath)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    encoder = LabelEncoder()
    df['ObesityCategory'] = encoder.fit_transform(df['ObesityCategory'])
    
    X = df[['Age', 'Gender', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']].values
    y = df['ObesityCategory'].values
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.35, random_state=42), encoder

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        self.layers = []
        self.biases = []
        self.dropout_rate = dropout_rate
        
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(np.random.randn(sizes[i], sizes[i+1]) * 0.01)
            self.biases.append(np.zeros((1, sizes[i+1])))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X, training=True):
        self.a = [X]
        for i in range(len(self.layers) - 1):
            X = self.relu(np.dot(X, self.layers[i]) + self.biases[i])
            if training:
                mask = (np.random.rand(*X.shape) > self.dropout_rate) / (1 - self.dropout_rate)
                X *= mask
            self.a.append(X)
        output = self.softmax(np.dot(X, self.layers[-1]) + self.biases[-1])
        self.a.append(output)
        return output
    
    def compute_loss(self, y_true, y_pred, loss_type="cross_entropy"):
        if loss_type == "cross_entropy":
            m = y_true.shape[0]
            return -np.sum(np.log(y_pred[np.arange(m), y_true] + 1e-8)) / m
        elif loss_type == "mse":
            y_one_hot = np.zeros_like(y_pred)
            y_one_hot[np.arange(y_true.shape[0]), y_true] = 1
            return np.mean((y_one_hot - y_pred) ** 2)
        return None
    
    def backward(self, X, y, learning_rate=0.01):
        m = X.shape[0]
        y_pred = self.a[-1]
        y_one_hot = np.zeros_like(y_pred)
        y_one_hot[np.arange(m), y] = 1
        
        dZ = y_pred - y_one_hot
        
        for i in range(len(self.layers) - 1, -1, -1):
            dW = np.dot(self.a[i].T, dZ) / m
            dB = np.sum(dZ, axis=0, keepdims=True) / m
            dZ = np.dot(dZ, self.layers[i].T) * (self.a[i] > 0)
            self.layers[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB
    
    def train(self, X_train, y_train, X_val, y_val, epochs=1000, batch_size=40, learning_rate=0.01, loss_type="cross_entropy", early_stopping=True, patience=10):
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        final_epoch = None
        
        for epoch in range(epochs):
            indices = np.random.permutation(X_train.shape[0])
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                y_batch = y_train_shuffled[i:i + batch_size]
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            
            y_train_pred = self.forward(X_train, training=False)
            y_val_pred = self.forward(X_val, training=False)
            
            train_loss = self.compute_loss(y_train, y_train_pred, loss_type)
            val_loss = self.compute_loss(y_val, y_val_pred, loss_type)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if early_stopping:
                        final_epoch = epoch
                        print(f"Early stopping triggered at epoch {epoch}")
                        break  
                    else:
                        if not final_epoch:
                            print('final_epoch', final_epoch)
                            final_epoch = epoch

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

        print("final epoch", final_epoch)
        
        epochs_range = range(len(train_losses)) 
        plt.plot(epochs_range, train_losses, label="Training Loss", color='blue')
        plt.plot(epochs_range, val_losses, label="Validation Loss", color='orange')
        plt.axvline(x=final_epoch, color='red', linestyle='--', label=f'stop at = {final_epoch}')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.show()

(filepath,) = ("obesity_data.csv",)  
(X_train, X_test, y_train, y_test), encoder = load_data(filepath)

# nn_3_layer = NeuralNetwork(input_size=6, hidden_sizes=[16, 16, 16], output_size=len(encoder.classes_))
# nn_4_layer = NeuralNetwork(input_size=6, hidden_sizes=[16, 16, 16, 16], output_size=len(encoder.classes_))

# print("Training 3-layer neural network...")
# nn_3_layer.train(X_train, y_train, X_test, y_test, loss_type="cross_entropy", early_stopping=False)

# print("Training 4-layer neural network...")
# nn_4_layer.train(X_train, y_train, X_test, y_test, loss_type="mse", early_stopping=False)

def train_nn():
    global encoder  # Ensure encoder is accessible
    hidden_layers = int(hidden_layers_var.get())
    neurons_per_layer = int(neurons_var.get())
    learning_rate = float(lr_var.get())
    batch_size = int(batch_size_var.get())
    epochs = int(epochs_var.get())
    loss_function = loss_var.get()
    early = early_var.get()
    
    hidden_sizes = [neurons_per_layer] * hidden_layers
    nn = NeuralNetwork(input_size=6, hidden_sizes=hidden_sizes, output_size=len(encoder.classes_))
    nn.train(X_train, y_train, X_test, y_test, epochs, batch_size, learning_rate, loss_function, early_stopping=early)

 
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

ttk.Button(root, text="Train", command=train_nn).grid(row=7, columnspan=2)
root.mainloop()