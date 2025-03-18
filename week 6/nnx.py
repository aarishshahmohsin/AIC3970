import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import seaborn as sns
import time
import os
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.weights = []
        self.biases = []
        self.initialize_weights()
        
    def initialize_weights(self):
        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        for i in range(1, len(layer_sizes)):
            
            scale = np.sqrt(2.0 / layer_sizes[i-1])
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i])))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        sigmoid_x = self.sigmoid(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X, training=True, dropout_rate=0, dropout_mask=None):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.relu(z)
            
            if training and dropout_rate > 0:
                if dropout_mask is None:
                    dropout_mask = np.random.binomial(1, 1-dropout_rate, size=a.shape) / (1-dropout_rate)
                a *= dropout_mask

            
            self.activations.append(a)
        
        
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        
        if self.output_size == 1:
            
            output = self.sigmoid(z)
        else:
            
            output = self.softmax(z)
        
        self.activations.append(output)
        return output
    
    def compute_loss(self, y_true, y_pred, l1_lambda=0, l2_lambda=0):
        m = y_true.shape[0]
        
        if self.output_size == 1:
            
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            total_loss = np.sum(loss) / m
        else:
            
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.sum(y_true * np.log(y_pred), axis=1)
            total_loss = np.sum(loss) / m
        
        
        if l2_lambda > 0:
            l2_reg = 0
            for w in self.weights:
                l2_reg += np.sum(np.square(w))
            total_loss += (l2_lambda / (2 * m)) * l2_reg
        
        
        if l1_lambda > 0:
            l1_reg = 0
            for w in self.weights:
                l1_reg += np.sum(np.abs(w))
            total_loss += (l1_lambda / m) * l1_reg
        
        return total_loss
    
    # def backward(self, X, y, learning_rate=0.01, l1_lambda=0, l2_lambda=0, dropout_rate=0):
    #     m = X.shape[0]
    #     gradients = []
        
    #     if self.output_size == 1:
            
    #         dZ = self.activations[-1] - y
    #     else:
            
    #         dZ = self.activations[-1] - y
        
    #     for layer in range(len(self.weights) - 1, -1, -1):
    #         dW = np.dot(self.activations[layer].T, dZ) / m
    #         db = np.sum(dZ, axis=0, keepdims=True) / m
            
            
    #         if l2_lambda > 0:
    #             dW += (l2_lambda / m) * self.weights[layer]
            
            
    #         if l1_lambda > 0:
    #             dW += (l1_lambda / m) * np.sign(self.weights[layer])
            
            
    #         gradients.insert(0, (dW, db))
            
            
    #         if layer > 0:
    #             dA = np.dot(dZ, self.weights[layer].T)
                
                
    #             if dropout_rate > 0:
    #                 dA *= (np.random.binomial(1, 1-dropout_rate, size=dA.shape) / (1-dropout_rate))
                
    #             if layer > 0:  
    #                 dZ = dA * self.relu_derivative(self.z_values[layer-1])
    #             else:  
    #                 dZ = dA * self.sigmoid_derivative(self.z_values[layer-1])
        

        
        # for i in range(len(self.weights)):
        #     self.weights[i] -= learning_rate * gradients[i][0]
        #     self.biases[i] -= learning_rate * gradients[i][1]

    # import numpy as np

    def backward(self, X, y, learning_rate=0.01, l1_lambda=0, l2_lambda=0, dropout_rate=0, beta1=0.9, beta2=0.999, epsilon=1e-8):
        m = X.shape[0]
        gradients = []

        if not hasattr(self, 'm_w'):
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
            self.t = 0  

        dZ = self.activations[-1] - y  

        for layer in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[layer].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            if l2_lambda > 0:
                dW += (l2_lambda / m) * self.weights[layer]
            if l1_lambda > 0:
                dW += (l1_lambda / m) * np.sign(self.weights[layer])

            gradients.insert(0, (dW, db))

            if layer > 0:
                dA = np.dot(dZ, self.weights[layer].T)

                if dropout_rate > 0:
                    dropout_mask = np.random.binomial(1, 1 - dropout_rate, size=dA.shape) / (1 - dropout_rate)
                    dA *= dropout_mask

                dZ = dA * self.relu_derivative(self.z_values[layer - 1]) if layer > 0 else dA * self.sigmoid_derivative(self.z_values[layer - 1])

        self.t += 1  
        for i in range(len(self.weights)):
            self.m_w[i] = beta1 * self.m_w[i] + (1 - beta1) * gradients[i][0]
            self.v_w[i] = beta2 * self.v_w[i] + (1 - beta2) * (gradients[i][0] ** 2)

            self.m_b[i] = beta1 * self.m_b[i] + (1 - beta1) * gradients[i][1]
            self.v_b[i] = beta2 * self.v_b[i] + (1 - beta2) * (gradients[i][1] ** 2)

            m_w_hat = self.m_w[i] / (1 - beta1 ** self.t)
            v_w_hat = self.v_w[i] / (1 - beta2 ** self.t)

            m_b_hat = self.m_b[i] / (1 - beta1 ** self.t)
            v_b_hat = self.v_b[i] / (1 - beta2 ** self.t)

            self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)

    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              learning_rate=0.01, l1_lambda=0, l2_lambda=0, dropout_rate=0, 
              verbose=True, early_stopping=True, patience=10):
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                
                self.forward(X_batch, training=True, dropout_rate=dropout_rate)
                
                
                self.backward(X_batch, y_batch, learning_rate, l1_lambda, l2_lambda, dropout_rate)
            
            
            y_train_pred = self.forward(X_train, training=False)
            train_loss = self.compute_loss(y_train, y_train_pred, l1_lambda, l2_lambda)
            train_accuracy = self.calculate_accuracy(y_train, y_train_pred)
            
            
            y_val_pred = self.forward(X_val, training=False)
            val_loss = self.compute_loss(y_val, y_val_pred, l1_lambda, l2_lambda)
            val_accuracy = self.calculate_accuracy(y_val, y_val_pred)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            epoch_time = time.time() - epoch_start_time
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Train Acc: {train_accuracy:.4f} - Val Acc: {val_accuracy:.4f}")
            
            
            
            
            
            
            
            
            
            
            
            
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def calculate_accuracy(self, y_true, y_pred):
        if self.output_size == 1:
            
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions == y_true)
        else:
            
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_true, axis=1)
            return np.mean(predictions == true_labels)
    
    def predict(self, X):
        y_pred = self.forward(X, training=False)
        
        if self.output_size == 1:
            
            return (y_pred > 0.5).astype(int)
        else:
            
            return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X, training=False)

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Regularization Visualizer")
        self.root.geometry("1200x800")
        
        
        self.dataset_var = tk.StringVar(value="breast_cancer")
        self.test_size_var = tk.DoubleVar(value=0.2)
        self.random_state_var = tk.IntVar(value=42)
        self.normalize_var = tk.BooleanVar(value=True)
        
        self.l1_var = tk.BooleanVar(value=False)
        self.l2_var = tk.BooleanVar(value=False)
        self.dropout_var = tk.BooleanVar(value=False)
        self.l1_lambda_var = tk.DoubleVar(value=0.001)
        self.l2_lambda_var = tk.DoubleVar(value=0.001)
        self.dropout_rate_var = tk.DoubleVar(value=0.2)
        
        self.epochs_var = tk.IntVar(value=100)
        self.batch_size_var = tk.IntVar(value=32)
        self.learning_rate_var = tk.DoubleVar(value=0.01)
        self.hidden_layers_var = tk.StringVar(value="10,10")
        
        
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        
        self.base_results = None
        self.reg_results = None
        
        
        self.create_widgets()
        
    def create_widgets(self):
        
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        
        config_frame = ttk.Frame(notebook)
        notebook.add(config_frame, text="Configuration")
        
        
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        
        
        self.setup_config_frame(config_frame)
        
        
        self.setup_results_frame(results_frame)
        
    def setup_config_frame(self, parent):
        
        dataset_frame = ttk.LabelFrame(parent, text="Dataset")
        dataset_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        
        ttk.Label(dataset_frame, text="Select Dataset:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        # datasets = ["breast_cancer", "iris", "wine", "digits", "custom"]
        datasets = ["breast_cancer"]
        dataset_menu = ttk.Combobox(dataset_frame, textvariable=self.dataset_var, values=datasets)
        dataset_menu.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # ttk.Button(dataset_frame, text="Browse Custom Dataset", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(dataset_frame, text="Test Size:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Scale(dataset_frame, from_=0.1, to=0.5, variable=self.test_size_var, orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(dataset_frame, textvariable=tk.StringVar(value=lambda: f"{self.test_size_var.get():.2f}")).grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(dataset_frame, text="Random State:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(dataset_frame, textvariable=self.random_state_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # ttk.Checkbutton(dataset_frame, text="Normalize Data", variable=self.normalize_var).grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)
        
        
        model_frame = ttk.LabelFrame(parent, text="Model Configuration")
        model_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(model_frame, text="Hidden Layers (comma separated):").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(model_frame, textvariable=self.hidden_layers_var, width=20).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(model_frame, text="Epochs:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(model_frame, textvariable=self.epochs_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(model_frame, text="Batch Size:").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(model_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(model_frame, text="Learning Rate:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(model_frame, textvariable=self.learning_rate_var, width=10).grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)
        
        
        reg_frame = ttk.LabelFrame(parent, text="Regularization")
        reg_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        
        ttk.Checkbutton(reg_frame, text="L1 Regularization", variable=self.l1_var, command=self.toggle_l1).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.l1_entry = ttk.Entry(reg_frame, textvariable=self.l1_lambda_var, width=10, state=tk.DISABLED)
        self.l1_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(reg_frame, text="Lambda").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        
        ttk.Checkbutton(reg_frame, text="L2 Regularization", variable=self.l2_var, command=self.toggle_l2).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.l2_entry = ttk.Entry(reg_frame, textvariable=self.l2_lambda_var, width=10, state=tk.DISABLED)
        self.l2_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(reg_frame, text="Lambda").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        
        
        ttk.Checkbutton(reg_frame, text="Dropout", variable=self.dropout_var, command=self.toggle_dropout).grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.dropout_entry = ttk.Entry(reg_frame, textvariable=self.dropout_rate_var, width=10, state=tk.DISABLED)
        self.dropout_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Label(reg_frame, text="Rate").grid(row=2, column=2, padx=5, pady=5, sticky=tk.W)
        
        
        run_frame = ttk.Frame(parent)
        run_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(run_frame, text="Load Data", command=self.load_data).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(run_frame, text="Train Models", command=self.train_models).grid(row=0, column=1, padx=5, pady=5)
        
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
    
    def setup_results_frame(self, parent):
        
        nav_frame = ttk.Frame(parent)
        nav_frame.pack(fill=tk.X, padx=10, pady=5)
        
        
        viz_notebook = ttk.Notebook(parent)
        viz_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        
        training_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(training_frame, text="Training Metrics")
        
        
        test_frame = ttk.Frame(viz_notebook)
        viz_notebook.add(test_frame, text="Test Metrics")
        
        
        self.setup_training_metrics_frame(training_frame)
        
        
        self.setup_test_metrics_frame(test_frame)
        
        
        ttk.Button(nav_frame, text="Refresh Visualizations", command=self.refresh_visualizations).pack(side=tk.LEFT, padx=5, pady=5)
    
    def setup_training_metrics_frame(self, parent):
        
        plots_frame = ttk.Frame(parent)
        plots_frame.pack(fill=tk.BOTH, expand=True)
        
        
        self.training_fig, self.training_axes = plt.subplots(2, 2, figsize=(6, 4))
        self.training_fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
        
        self.training_axes[0, 0].set_title("Training Loss")
        self.training_axes[0, 0].set_xlabel("Epoch")
        self.training_axes[0, 0].set_ylabel("Loss")
        
        self.training_axes[0, 1].set_title("Validation Loss")
        self.training_axes[0, 1].set_xlabel("Epoch")
        self.training_axes[0, 1].set_ylabel("Loss")
        
        self.training_axes[1, 0].set_title("Training Accuracy")
        self.training_axes[1, 0].set_xlabel("Epoch")
        self.training_axes[1, 0].set_ylabel("Accuracy")
        
        self.training_axes[1, 1].set_title("Validation Accuracy")
        self.training_axes[1, 1].set_xlabel("Epoch")
        self.training_axes[1, 1].set_ylabel("Accuracy")
        
        
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, master=plots_frame)
        self.training_canvas.draw()
        self.training_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_test_metrics_frame(self, parent):
        
        test_notebook = ttk.Notebook(parent)
        test_notebook.pack(fill=tk.BOTH, expand=True)
        
        
        cm_frame = ttk.Frame(test_notebook)
        test_notebook.add(cm_frame, text="Confusion Matrix")
        
        
        roc_frame = ttk.Frame(test_notebook)
        test_notebook.add(roc_frame, text="ROC Curve")
        
        
        metrics_frame = ttk.Frame(test_notebook)
        test_notebook.add(metrics_frame, text="Metrics Table")
        
        
        self.setup_confusion_matrix_frame(cm_frame)
        
        
        self.setup_roc_curve_frame(roc_frame)
        
        
        self.setup_metrics_table_frame(metrics_frame)
    
    def setup_confusion_matrix_frame(self, parent):
        
        plots_frame = ttk.Frame(parent)
        plots_frame.pack(fill=tk.BOTH, expand=True)
        
        
        self.cm_fig, self.cm_axes = plt.subplots(1, 2, figsize=(6, 4))
        self.cm_fig.subplots_adjust(wspace=0.4)
        
        
        self.cm_axes[0].set_title("Confusion Matrix (No Regularization)")
        self.cm_axes[1].set_title("Confusion Matrix (With Regularization)")
        
       
        self.cm_canvas = FigureCanvasTkAgg(self.cm_fig, master=plots_frame)
        self.cm_canvas.draw()
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_roc_curve_frame(self, parent):
        
        plots_frame = ttk.Frame(parent)
        plots_frame.pack(fill=tk.BOTH, expand=True)
        
        
        self.roc_fig, self.roc_ax = plt.subplots(figsize=(6, 4))
        
        
        self.roc_ax.set_title("ROC Curve Comparison")
        self.roc_ax.set_xlabel("False Positive Rate")
        self.roc_ax.set_ylabel("True Positive Rate")
        self.roc_ax.plot([0, 1], [0, 1], 'k--')
        self.roc_ax.set_xlim([0.0, 1.0])
        self.roc_ax.set_ylim([0.0, 1.05])
        self.roc_ax.grid(True)
        
        
        self.roc_canvas = FigureCanvasTkAgg(self.roc_fig, master=plots_frame)
        self.roc_canvas.draw()
        self.roc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_metrics_table_frame(self, parent):
        
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        
        columns = ("Metric", "No Regularization", "With Regularization", "Improvement")
        self.metrics_tree = ttk.Treeview(table_frame, columns=columns, show="headings")
        
        
        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=150, anchor=tk.CENTER)
        
        
        metrics = ["F1 Score", "Precision", "Recall", "ROC AUC"]
        for metric in metrics:
            self.metrics_tree.insert("", tk.END, values=(metric, "-", "-", "-"))
        
        
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=scrollbar.set)
        
        
        self.metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def toggle_l1(self):
        if self.l1_var.get():
            self.l1_entry.configure(state=tk.NORMAL)
        else:
            self.l1_entry.configure(state=tk.DISABLED)
    
    def toggle_l2(self):
        if self.l2_var.get():
            self.l2_entry.configure(state=tk.NORMAL)
        else:
            self.l2_entry.configure(state=tk.DISABLED)
    
    def toggle_dropout(self):
        if self.dropout_var.get():
            self.dropout_entry.configure(state=tk.NORMAL)
        else:
            self.dropout_entry.configure(state=tk.DISABLED)
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=(("CSV Files", "*.csv"), ("Excel Files", "*.xlsx"), ("All Files", "*.*"))
        )
        if filename:
            self.dataset_var.set("custom")
            self.custom_file_path = filename
            self.status_var.set(f"Selected file: {os.path.basename(filename)}")
    
    def load_data(self):
        try:
            self.status_var.set("Loading dataset...")
            self.root.update()
            
            dataset_name = self.dataset_var.get()
            
            if dataset_name == "breast_cancer":
                data = load_breast_cancer()
                X = data.data
                y = data.target
                self.multi_class = False
                
            elif dataset_name == "iris":
                data = load_iris()
                X = data.data
                y = data.target
                self.multi_class = True
                
            elif dataset_name == "wine":
                data = load_wine()
                X = data.data
                y = data.target
                self.multi_class = True
                
            elif dataset_name == "digits":
                data = load_digits()
                X = data.data
                y = data.target
                self.multi_class = True
                
            elif dataset_name == "custom":
                if not hasattr(self, 'custom_file_path'):
                    messagebox.showerror("Error", "Please select a custom dataset file first.")
                    return
                
                
                if self.custom_file_path.endswith('.csv'):
                    df = pd.read_csv(self.custom_file_path)
                elif self.custom_file_path.endswith('.xlsx'):
                    df = pd.read_excel(self.custom_file_path)
                else:
                    messagebox.showerror("Error", "Unsupported file format.")
                    return
                
                
                target_selector = tk.Toplevel(self.root)
                target_selector.title("Select Target Column")
                target_selector.geometry("300x200")
                
                ttk.Label(target_selector, text="Select target column:").pack(pady=10)
                
                target_var = tk.StringVar()
                target_combo = ttk.Combobox(target_selector, textvariable=target_var, values=df.columns.tolist())
                target_combo.pack(pady=10)
                
                multi_class_var = tk.BooleanVar(value=False)
                ttk.Checkbutton(target_selector, text="Multi-class classification", variable=multi_class_var).pack(pady=10)
                
                def confirm_target():
                    nonlocal X, y
                    target_col = target_var.get()
                    if not target_col:
                        messagebox.showerror("Error", "Please select a target column.")
                        return
                    
                    
                    X = df.drop(columns=[target_col]).values
                    y_raw = df[target_col].values
                    
                    
                    self.multi_class = multi_class_var.get()
                    if self.multi_class:
                        le = LabelEncoder()
                        y = le.fit_transform(y_raw)
                    else:
                        y = y_raw
                    
                    target_selector.destroy()
                
                ttk.Button(target_selector, text="Confirm", command=confirm_target).pack(pady=10)
                
                
                self.root.wait_window(target_selector)
                
                if 'y' not in locals():
                    return  
            
            else:
                messagebox.showerror("Error", "Invalid dataset selection.")
                return
            
            
            X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(
                X, y, test_size=self.test_size_var.get(), random_state=self.random_state_var.get()
            )
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=self.random_state_var.get()
            )
            
            
            if self.normalize_var.get():
                scaler = StandardScaler()
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_val = scaler.transform(self.X_val)
                self.X_test = scaler.transform(self.X_test)
            
            
            if self.multi_class:
                num_classes = len(np.unique(y))
                self.y_train_onehot = np.eye(num_classes)[self.y_train]
                self.y_val_onehot = np.eye(num_classes)[self.y_val]
                self.y_test_onehot = np.eye(num_classes)[self.y_test]
                self.output_size = num_classes
            else:
                
                self.y_train_onehot = self.y_train.reshape(-1, 1)
                self.y_val_onehot = self.y_val.reshape(-1, 1)
                self.y_test_onehot = self.y_test.reshape(-1, 1)
                self.output_size = 1
            
            
            self.input_size = self.X_train.shape[1]
            
            self.status_var.set(f"Data loaded successfully. Features: {self.input_size}, Classes: {self.output_size}")
            
            messagebox.showinfo("Success", "Dataset loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
            self.status_var.set("Error loading dataset")
    
    def train_models(self):
        if not hasattr(self, 'X_train') or self.X_train is None:
            messagebox.showerror("Error", "Please load a dataset first.")
            return
        
        try:
            
            hidden_layers_str = self.hidden_layers_var.get()
            try:
                hidden_sizes = [int(x.strip()) for x in hidden_layers_str.split(',')]
            except ValueError:
                messagebox.showerror("Error", "Invalid hidden layers format. Use comma-separated integers.")
                return
            
            
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            learning_rate = self.learning_rate_var.get()
            
            
            self.status_var.set("Training base model (no regularization)...")
            self.root.update()
            
            base_model = NeuralNetwork(self.input_size, hidden_sizes, self.output_size)
            
            base_train_losses, base_val_losses, base_train_accs, base_val_accs = base_model.train(
                self.X_train, self.y_train_onehot,
                self.X_val, self.y_val_onehot,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                l1_lambda=0.0,
                l2_lambda=0.0,
                dropout_rate=0.0,
                verbose=True
            )
            
            
            base_y_test_pred = base_model.predict_proba(self.X_test)
            
            
            self.status_var.set("Training regularized model...")
            self.root.update()
            
            
            l1_lambda = self.l1_lambda_var.get() if self.l1_var.get() else 0.0
            l2_lambda = self.l2_lambda_var.get() if self.l2_var.get() else 0.0
            dropout_rate = self.dropout_rate_var.get() if self.dropout_var.get() else 0.0
            
            reg_model = NeuralNetwork(self.input_size, hidden_sizes, self.output_size)
            
            reg_train_losses, reg_val_losses, reg_train_accs, reg_val_accs = reg_model.train(
                self.X_train, self.y_train_onehot,
                self.X_val, self.y_val_onehot,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                l1_lambda=l1_lambda,
                l2_lambda=l2_lambda,
                dropout_rate=dropout_rate,
                verbose=True
            )
            
            
            reg_y_test_pred = reg_model.predict_proba(self.X_test)
            
            
            self.base_results = {
                'model': base_model,
                'train_losses': base_train_losses,
                'val_losses': base_val_losses,
                'train_accs': base_train_accs,
                'val_accs': base_val_accs,
                'y_test_pred': base_y_test_pred
            }
            
            self.reg_results = {
                'model': reg_model,
                'train_losses': reg_train_losses,
                'val_losses': reg_val_losses,
                'train_accs': reg_train_accs,
                'val_accs': reg_val_accs,
                'y_test_pred': reg_y_test_pred,
                'l1_lambda': l1_lambda,
                'l2_lambda': l2_lambda,
                'dropout_rate': dropout_rate
            }
            
            
            self.refresh_visualizations()
            
            
            reg_msg = []
            if l1_lambda > 0:
                reg_msg.append(f"L1 (λ={l1_lambda})")
            if l2_lambda > 0:
                reg_msg.append(f"L2 (λ={l2_lambda})")
            if dropout_rate > 0:
                reg_msg.append(f"Dropout (rate={dropout_rate})")
            
            reg_type = ", ".join(reg_msg) if reg_msg else "None"
            
            messagebox.showinfo("Success", f"Models trained successfully!\nBase model: No regularization\nRegularized model: {reg_type}")
            
            self.status_var.set("Models trained successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training models: {str(e)}")
            self.status_var.set("Error training models")
    
    def refresh_visualizations(self):
        if not hasattr(self, 'base_results') or self.base_results is None:
            messagebox.showinfo("Info", "Please train models first.")
            return
        
        try:
            self.status_var.set("Refreshing visualizations...")
            self.root.update()
            
            
            self.update_training_metrics()
            
            
            self.update_test_metrics()
            
            self.status_var.set("Visualizations refreshed successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error refreshing visualizations: {str(e)}")
            self.status_var.set("Error refreshing visualizations")

    def update_training_metrics(self):
        
        for ax in self.training_axes.flatten():
            ax.clear()
        
        
        self.training_axes[0, 0].plot(self.base_results['train_losses'], label='No Regularization')
        self.training_axes[0, 0].plot(self.reg_results['train_losses'], label='With Regularization')
        self.training_axes[0, 0].set_title("Training Loss")
        self.training_axes[0, 0].set_xlabel("Epoch")
        self.training_axes[0, 0].set_ylabel("Loss")
        self.training_axes[0, 0].legend()
        self.training_axes[0, 0].grid(True)
        
        
        self.training_axes[0, 1].plot(self.base_results['val_losses'], label='No Regularization')
        self.training_axes[0, 1].plot(self.reg_results['val_losses'], label='With Regularization')
        self.training_axes[0, 1].set_title("Validation Loss")
        self.training_axes[0, 1].set_xlabel("Epoch")
        self.training_axes[0, 1].set_ylabel("Loss")
        self.training_axes[0, 1].legend()
        self.training_axes[0, 1].grid(True)
        
        
        self.training_axes[1, 0].plot(self.base_results['train_accs'], label='No Regularization')
        self.training_axes[1, 0].plot(self.reg_results['train_accs'], label='With Regularization')
        self.training_axes[1, 0].set_title("Training Accuracy")
        self.training_axes[1, 0].set_xlabel("Epoch")
        self.training_axes[1, 0].set_ylabel("Accuracy")
        self.training_axes[1, 0].legend()
        self.training_axes[1, 0].grid(True)
        
        
        self.training_axes[1, 1].plot(self.base_results['val_accs'], label='No Regularization')
        self.training_axes[1, 1].plot(self.reg_results['val_accs'], label='With Regularization')
        self.training_axes[1, 1].set_title("Validation Accuracy")
        self.training_axes[1, 1].set_xlabel("Epoch")
        self.training_axes[1, 1].set_ylabel("Accuracy")
        self.training_axes[1, 1].legend()
        self.training_axes[1, 1].grid(True)
        
        
        self.training_fig.tight_layout()
        self.training_canvas.draw()
    
    def update_test_metrics(self):
        
        base_y_pred = self.base_results['y_test_pred']
        reg_y_pred = self.reg_results['y_test_pred']
        
        
        if self.output_size == 1:
            base_y_pred_class = (base_y_pred > 0.5).astype(int)
            reg_y_pred_class = (reg_y_pred > 0.5).astype(int)
            
            
            base_cm = confusion_matrix(self.y_test, base_y_pred_class)
            reg_cm = confusion_matrix(self.y_test, reg_y_pred_class)
            
            
            for ax in self.cm_axes:
                ax.clear()
            
            sns.heatmap(base_cm, annot=True, fmt='d', cmap='Blues', ax=self.cm_axes[0])
            self.cm_axes[0].set_title("Confusion Matrix (No Regularization)")
            self.cm_axes[0].set_xlabel("Predicted Label")
            self.cm_axes[0].set_ylabel("True Label")
            
            sns.heatmap(reg_cm, annot=True, fmt='d', cmap='Blues', ax=self.cm_axes[1])
            self.cm_axes[1].set_title("Confusion Matrix (With Regularization)")
            self.cm_axes[1].set_xlabel("Predicted Label")
            self.cm_axes[1].set_ylabel("True Label")
            
            self.cm_fig.tight_layout()
            self.cm_canvas.draw()
            
            
            base_f1 = f1_score(self.y_test, base_y_pred_class)
            reg_f1 = f1_score(self.y_test, reg_y_pred_class)
            
            base_precision = precision_score(self.y_test, base_y_pred_class)
            reg_precision = precision_score(self.y_test, reg_y_pred_class)
            
            base_recall = recall_score(self.y_test, base_y_pred_class)
            reg_recall = recall_score(self.y_test, reg_y_pred_class)
            
            base_roc_auc = roc_auc_score(self.y_test, base_y_pred)
            reg_roc_auc = roc_auc_score(self.y_test, reg_y_pred)
            
            
            metrics = [
                ("F1 Score", base_f1, reg_f1, reg_f1 - base_f1),
                ("Precision", base_precision, reg_precision, reg_precision - base_precision),
                ("Recall", base_recall, reg_recall, reg_recall - base_recall),
                ("ROC AUC", base_roc_auc, reg_roc_auc, reg_roc_auc - base_roc_auc)
            ]
            
            
            for item in self.metrics_tree.get_children():
                self.metrics_tree.delete(item)
            
            
            for i, (metric, base, reg, diff) in enumerate(metrics):
                color = "green" if diff > 0 else ("red" if diff < 0 else "black")
                self.metrics_tree.insert("", i, values=(
                    metric, 
                    f"{base:.4f}", 
                    f"{reg:.4f}", 
                    f"{diff:.4f}"
                ))
            
            
            self.roc_ax.clear()
            
            
            base_fpr, base_tpr, _ = roc_curve(self.y_test, base_y_pred)
            reg_fpr, reg_tpr, _ = roc_curve(self.y_test, reg_y_pred)
            
            
            self.roc_ax.plot(base_fpr, base_tpr, label=f"No Regularization (AUC = {base_roc_auc:.4f})")
            self.roc_ax.plot(reg_fpr, reg_tpr, label=f"With Regularization (AUC = {reg_roc_auc:.4f})")
            self.roc_ax.plot([0, 1], [0, 1], 'k--')
            self.roc_ax.set_title("ROC Curve Comparison")
            self.roc_ax.set_xlabel("False Positive Rate")
            self.roc_ax.set_ylabel("True Positive Rate")
            self.roc_ax.set_xlim([0.0, 1.0])
            self.roc_ax.set_ylim([0.0, 1.05])
            self.roc_ax.legend(loc="lower right")
            self.roc_ax.grid(True)
            
            self.roc_fig.tight_layout()
            self.roc_canvas.draw()
            
        else:
            
            base_y_pred_class = np.argmax(base_y_pred, axis=1)
            reg_y_pred_class = np.argmax(reg_y_pred, axis=1)
            
            
            base_cm = confusion_matrix(self.y_test, base_y_pred_class)
            reg_cm = confusion_matrix(self.y_test, reg_y_pred_class)
            
            
            for ax in self.cm_axes:
                ax.clear()
            
            sns.heatmap(base_cm, annot=True, fmt='d', cmap='Blues', ax=self.cm_axes[0])
            self.cm_axes[0].set_title("Confusion Matrix (No Regularization)")
            self.cm_axes[0].set_xlabel("Predicted Label")
            self.cm_axes[0].set_ylabel("True Label")
            
            sns.heatmap(reg_cm, annot=True, fmt='d', cmap='Blues', ax=self.cm_axes[1])
            self.cm_axes[1].set_title("Confusion Matrix (With Regularization)")
            self.cm_axes[1].set_xlabel("Predicted Label")
            self.cm_axes[1].set_ylabel("True Label")
            
            self.cm_fig.tight_layout()
            self.cm_canvas.draw()
            
            
            base_f1 = f1_score(self.y_test, base_y_pred_class, average='weighted')
            reg_f1 = f1_score(self.y_test, reg_y_pred_class, average='weighted')
            
            base_precision = precision_score(self.y_test, base_y_pred_class, average='weighted')
            reg_precision = precision_score(self.y_test, reg_y_pred_class, average='weighted')
            
            base_recall = recall_score(self.y_test, base_y_pred_class, average='weighted')
            reg_recall = recall_score(self.y_test, reg_y_pred_class, average='weighted')
            
            
            base_roc_auc = roc_auc_score(self.y_test_onehot, base_y_pred, average='weighted', multi_class='ovr')
            reg_roc_auc = roc_auc_score(self.y_test_onehot, reg_y_pred, average='weighted', multi_class='ovr')
            
            
            metrics = [
                ("F1 Score", base_f1, reg_f1, reg_f1 - base_f1),
                ("Precision", base_precision, reg_precision, reg_precision - base_precision),
                ("Recall", base_recall, reg_recall, reg_recall - base_recall),
                ("ROC AUC", base_roc_auc, reg_roc_auc, reg_roc_auc - base_roc_auc)
            ]
            
            
            for item in self.metrics_tree.get_children():
                self.metrics_tree.delete(item)
            
            
            for i, (metric, base, reg, diff) in enumerate(metrics):
                color = "green" if diff > 0 else ("red" if diff < 0 else "black")
                self.metrics_tree.insert("", i, values=(
                    metric, 
                    f"{base:.4f}", 
                    f"{reg:.4f}", 
                    f"{diff:.4f}"
                ))
            
            
            self.roc_ax.clear()
            
            
            classes = np.unique(self.y_test)
            base_aucs = []
            reg_aucs = []
            
            for i in range(len(classes)):
                y_true_bin = (self.y_test == i).astype(int)
                base_auc = roc_auc_score(y_true_bin, base_y_pred[:, i])
                reg_auc = roc_auc_score(y_true_bin, reg_y_pred[:, i])
                base_aucs.append(base_auc)
                reg_aucs.append(reg_auc)
            
            
            x = np.arange(len(classes))
            width = 0.35
            
            self.roc_ax.bar(x - width/2, base_aucs, width, label='No Regularization')
            self.roc_ax.bar(x + width/2, reg_aucs, width, label='With Regularization')
            
            self.roc_ax.set_title("Class-wise ROC AUC Scores")
            self.roc_ax.set_xlabel("Class")
            self.roc_ax.set_ylabel("ROC AUC Score")
            self.roc_ax.set_xticks(x)
            self.roc_ax.set_xticklabels([f"Class {i}" for i in classes])
            self.roc_ax.legend()
            self.roc_ax.grid(True, axis='y')
            
            self.roc_fig.tight_layout()
            self.roc_canvas.draw()


if __name__ == "__main__":
    plt.rcParams['font.size'] = 7  
    plt.rcParams['axes.titlesize'] = 7  
    plt.rcParams['axes.labelsize'] = 7  
    plt.rcParams['xtick.labelsize'] =7  
    plt.rcParams['ytick.labelsize'] = 7  
    plt.rcParams['legend.fontsize'] = 7  

    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()