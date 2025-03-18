import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import seaborn as sns
import time

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
            # He initialization
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
        
        # Output layer
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        
        if self.output_size == 1:
            # Binary classification
            output = self.sigmoid(z)
        else:
            # Multi-class classification
            output = self.softmax(z)
        
        self.activations.append(output)
        return output
    
    def compute_loss(self, y_true, y_pred, l1_lambda=0, l2_lambda=0):
        m = y_true.shape[0]
        
        if self.output_size == 1:
            # Binary cross-entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            total_loss = np.sum(loss) / m
        else:
            # Categorical cross-entropy
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.sum(y_true * np.log(y_pred), axis=1)
            total_loss = np.sum(loss) / m
        
        # L2 regularization
        if l2_lambda > 0:
            l2_reg = 0
            for w in self.weights:
                l2_reg += np.sum(np.square(w))
            total_loss += (l2_lambda / (2 * m)) * l2_reg
        
        # L1 regularization
        if l1_lambda > 0:
            l1_reg = 0
            for w in self.weights:
                l1_reg += np.sum(np.abs(w))
            total_loss += (l1_lambda / m) * l1_reg
        
        return total_loss
    
    def backward(self, X, y, learning_rate=0.01, l1_lambda=0, l2_lambda=0, dropout_rate=0):
        m = X.shape[0]
        gradients = []
        
        if self.output_size == 1:
            # Binary classification
            dZ = self.activations[-1] - y
        else:
            # Multi-class classification
            dZ = self.activations[-1] - y
        
        for layer in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(self.activations[layer].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Apply L2 regularization to weights
            if l2_lambda > 0:
                dW += (l2_lambda / m) * self.weights[layer]
            
            # Apply L1 regularization to weights
            if l1_lambda > 0:
                dW += (l1_lambda / m) * np.sign(self.weights[layer])
            
            # Store gradients
            gradients.insert(0, (dW, db))
            
            # Continue backpropagation if not at input layer
            if layer > 0:
                dA = np.dot(dZ, self.weights[layer].T)
                
                # Apply dropout mask if needed
                if dropout_rate > 0:
                    dA *= (np.random.binomial(1, 1-dropout_rate, size=dA.shape) / (1-dropout_rate))
                
                if layer > 0:  # ReLU for hidden layers
                    dZ = dA * self.relu_derivative(self.z_values[layer-1])
                else:  # Sigmoid for input layer
                    dZ = dA * self.sigmoid_derivative(self.z_values[layer-1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients[i][0]
            self.biases[i] -= learning_rate * gradients[i][1]
    
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
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                self.forward(X_batch, training=True, dropout_rate=dropout_rate)
                
                # Backward pass
                self.backward(X_batch, y_batch, learning_rate, l1_lambda, l2_lambda, dropout_rate)
            
            # Compute training loss and accuracy
            y_train_pred = self.forward(X_train, training=False)
            train_loss = self.compute_loss(y_train, y_train_pred, l1_lambda, l2_lambda)
            train_accuracy = self.calculate_accuracy(y_train, y_train_pred)
            
            # Compute validation loss and accuracy
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
            
            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def calculate_accuracy(self, y_true, y_pred):
        if self.output_size == 1:
            # Binary classification
            predictions = (y_pred > 0.5).astype(int)
            return np.mean(predictions == y_true)
        else:
            # Multi-class classification
            predictions = np.argmax(y_pred, axis=1)
            true_labels = np.argmax(y_true, axis=1)
            return np.mean(predictions == true_labels)
    
    def predict(self, X):
        y_pred = self.forward(X, training=False)
        
        if self.output_size == 1:
            # Binary classification
            return (y_pred > 0.5).astype(int)
        else:
            # Multi-class classification
            return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X, training=False)


class NeuralNetworkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network with Regularization Techniques")
        self.root.geometry("1200x800")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for controls
        left_panel = ttk.Frame(main_frame, padding="10", relief=tk.RAISED)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Create right panel for visualizations
        self.right_panel = ttk.Frame(main_frame, padding="10", relief=tk.RAISED)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add controls to left panel
        ttk.Label(left_panel, text="Dataset", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        
        # Dataset selection
        self.dataset_var = tk.StringVar(value="Iris")
        ttk.Radiobutton(left_panel, text="Iris", variable=self.dataset_var, value="Iris").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(left_panel, text="Breast Cancer", variable=self.dataset_var, value="Breast Cancer").grid(row=2, column=0, sticky=tk.W)
        ttk.Radiobutton(left_panel, text="Wine", variable=self.dataset_var, value="Wine").grid(row=3, column=0, sticky=tk.W)
        
        # Network architecture
        ttk.Label(left_panel, text="Network Architecture", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky=tk.W, pady=5)
        
        self.hidden_layers_var = tk.StringVar(value="32,16")
        ttk.Label(left_panel, text="Hidden Layers (comma-separated):").grid(row=5, column=0, sticky=tk.W)
        ttk.Entry(left_panel, textvariable=self.hidden_layers_var, width=20).grid(row=6, column=0, sticky=tk.W, pady=2)
        
        # Training parameters
        ttk.Label(left_panel, text="Training Parameters", font=("Arial", 12, "bold")).grid(row=7, column=0, sticky=tk.W, pady=5)
        
        self.epochs_var = tk.IntVar(value=100)
        ttk.Label(left_panel, text="Epochs:").grid(row=8, column=0, sticky=tk.W)
        ttk.Entry(left_panel, textvariable=self.epochs_var, width=10).grid(row=9, column=0, sticky=tk.W, pady=2)
        
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Label(left_panel, text="Batch Size:").grid(row=10, column=0, sticky=tk.W)
        ttk.Entry(left_panel, textvariable=self.batch_size_var, width=10).grid(row=11, column=0, sticky=tk.W, pady=2)
        
        self.learning_rate_var = tk.DoubleVar(value=0.01)
        ttk.Label(left_panel, text="Learning Rate:").grid(row=12, column=0, sticky=tk.W)
        ttk.Entry(left_panel, textvariable=self.learning_rate_var, width=10).grid(row=13, column=0, sticky=tk.W, pady=2)
        
        # Regularization techniques
        ttk.Label(left_panel, text="Regularization Techniques", font=("Arial", 12, "bold")).grid(row=14, column=0, sticky=tk.W, pady=5)
        
        self.l1_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_panel, text="L1 Regularization", variable=self.l1_var).grid(row=15, column=0, sticky=tk.W)
        
        self.l1_lambda_var = tk.DoubleVar(value=0.001)
        ttk.Label(left_panel, text="L1 Lambda:").grid(row=16, column=0, sticky=tk.W)
        ttk.Entry(left_panel, textvariable=self.l1_lambda_var, width=10).grid(row=17, column=0, sticky=tk.W, pady=2)
        
        self.l2_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_panel, text="L2 Regularization", variable=self.l2_var).grid(row=18, column=0, sticky=tk.W)
        
        self.l2_lambda_var = tk.DoubleVar(value=0.001)
        ttk.Label(left_panel, text="L2 Lambda:").grid(row=19, column=0, sticky=tk.W)
        ttk.Entry(left_panel, textvariable=self.l2_lambda_var, width=10).grid(row=20, column=0, sticky=tk.W, pady=2)
        
        self.dropout_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_panel, text="Dropout", variable=self.dropout_var).grid(row=21, column=0, sticky=tk.W)
        
        self.dropout_rate_var = tk.DoubleVar(value=0.3)
        ttk.Label(left_panel, text="Dropout Rate:").grid(row=22, column=0, sticky=tk.W)
        ttk.Entry(left_panel, textvariable=self.dropout_rate_var, width=10).grid(row=23, column=0, sticky=tk.W, pady=2)
        
        # Buttons
        buttons_frame = ttk.Frame(left_panel)
        buttons_frame.grid(row=24, column=0, sticky=tk.W, pady=10)
        
        ttk.Button(buttons_frame, text="Train Without Regularization", command=self.train_without_regularization).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Train With Regularization", command=self.train_with_regularization).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left_panel, textvariable=self.status_var, font=("Arial", 10), relief=tk.SUNKEN).grid(row=25, column=0, sticky=tk.W+tk.E, pady=10)
        
        # Initialize data and model
        self.initialize_data()
        self.create_visualization_tabs()
        
    def initialize_data(self):
        self.status_var.set("Loading dataset...")
        self.root.update()
        
        dataset_name = self.dataset_var.get()
        
        # Generate or load dataset
        if dataset_name == "Iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            X = data.data
            y = data.target
            self.class_names = data.target_names
            self.feature_names = data.feature_names
            self.num_classes = len(np.unique(y))
        
        elif dataset_name == "Breast Cancer":
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            X = data.data
            y = data.target
            self.class_names = data.target_names
            self.feature_names = data.feature_names
            self.num_classes = len(np.unique(y))
        
        elif dataset_name == "Wine":
            from sklearn.datasets import load_wine
            data = load_wine()
            X = data.data
            y = data.target
            self.class_names = data.target_names
            self.feature_names = data.feature_names
            self.num_classes = len(np.unique(y))
        
        # Split data
        X_train_val, self.X_test, y_train_val, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)
        
        # Scale data
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.X_test = scaler.transform(self.X_test)
        
        # One-hot encode target for multiclass
        if self.num_classes > 2:
            self.y_train_onehot = np.eye(self.num_classes)[self.y_train]
            self.y_val_onehot = np.eye(self.num_classes)[self.y_val]
            self.y_test_onehot = np.eye(self.num_classes)[self.y_test]
            self.output_size = self.num_classes
        else:
            # Binary classification
            self.y_train_onehot = self.y_train.reshape(-1, 1)
            self.y_val_onehot = self.y_val.reshape(-1, 1)
            self.y_test_onehot = self.y_test.reshape(-1, 1)
            self.output_size = 1
        
        self.input_size = self.X_train.shape[1]
        
        self.status_var.set(f"Loaded {dataset_name} dataset with {self.input_size} features and {self.num_classes} classes")
        
    def create_model(self):
        # Parse hidden layers
        hidden_layers_str = self.hidden_layers_var.get()
        hidden_sizes = [int(x.strip()) for x in hidden_layers_str.split(',')]
        
        # Create model
        model = NeuralNetwork(self.input_size, hidden_sizes, self.output_size)
        return model
    
    def create_visualization_tabs(self):
        # Clear existing tabs
        for widget in self.right_panel.winfo_children():
            widget.destroy()
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.loss_tab = ttk.Frame(self.notebook)
        self.accuracy_tab = ttk.Frame(self.notebook)
        self.confusion_matrix_tab = ttk.Frame(self.notebook)
        self.metrics_tab = ttk.Frame(self.notebook)
        self.roc_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.loss_tab, text="Loss")
        self.notebook.add(self.accuracy_tab, text="Accuracy")
        self.notebook.add(self.confusion_matrix_tab, text="Confusion Matrix")
        self.notebook.add(self.metrics_tab, text="Metrics")
        self.notebook.add(self.roc_tab, text="ROC Curve")
        
        # Create figures and axes for plots
        self.loss_fig, self.loss_ax = plt.subplots(figsize=(8, 6))
        self.accuracy_fig, self.accuracy_ax = plt.subplots(figsize=(8, 6))
        self.cm_fig, self.cm_ax = plt.subplots(figsize=(8, 6))
        self.metrics_fig, self.metrics_ax = plt.subplots(figsize=(8, 6))
        self.roc_fig, self.roc_ax = plt.subplots(figsize=(8, 6))
        
        # Create canvas for each figure
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, self.loss_tab)
        self.loss_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.accuracy_canvas = FigureCanvasTkAgg(self.accuracy_fig, self.accuracy_tab)
        self.accuracy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.cm_canvas = FigureCanvasTkAgg(self.cm_fig, self.confusion_matrix_tab)
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, self.metrics_tab)
        self.metrics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.roc_canvas = FigureCanvasTkAgg(self.roc_fig, self.roc_tab)
        self.roc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def train_without_regularization(self):
        self.status_var.set("Training without regularization...")
        self.root.update()
        
        # Create model
        self.model_noreg = self.create_model()
        
        # Train model
        epochs = self.epochs_var.get()
        batch_size = self.batch_size_var.get()
        learning_rate = self.learning_rate_var.get()
        
        train_losses, val_losses, train_accs, val_accs = self.model_noreg.train(
            self.X_train, self.y_train_onehot,
            self.X_val, self.y_val_onehot,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            verbose=True
        )
        
        self.noreg_results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs
        }
        
        # Evaluate on test set
        y_pred = self.model_noreg.predict(self.X_test)
        y_pred_proba = self.model_noreg.predict_proba(self.X_test)
        
        if self.output_size == 1:
            cm = confusion_matrix(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            
            self.noreg_metrics = {
                'confusion_matrix': cm,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr
            }
        else:
            # For multiclass
            y_true = self.y_test
            cm = confusion_matrix(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            
            # Calculate one-vs-rest ROC AUC
            roc_auc = roc_auc_score(self.y_test_onehot, y_pred_proba, multi_class='ovr', average='macro')
            
            self.noreg_metrics = {
                'confusion_matrix': cm,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc
            }
        
        # Update plots
        self.update_plots()
        
        self.status_var.set("Training without regularization complete")
    
    def train_with_regularization(self):
        self.status_var.set("Training with regularization...")
        self.root.update()
        
        # Get regularization parameters
        l1_lambda = self.l1_lambda_var.get() if self.l1_var.get() else 0.0
        l2_lambda = self.l2_lambda_var.get() if self.l2_var.get() else 0.0
        dropout_rate = self.dropout_rate_var.get() if self.dropout_var.get() else 0.0
        
        reg_types = []
        if l1_lambda > 0:
            reg_types.append("L1")
        if l2_lambda > 0:
            reg_types.append("L2")
        if dropout_rate > 0:
            reg_types.append("Dropout")
        
        reg_description = " + ".join(reg_types) if reg_types else "No regularization"
        
        # Create model
        self.model_reg = self.create_model()
        
        # Train model
        epochs = self.epochs_var.get()
        batch_size = self.batch_size_var.get()
        learning_rate = self.learning_rate_var.get()
        
        train_losses, val_losses, train_accs, val_accs = self.model_reg.train(
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
        
        self.reg_results = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'description': reg_description
        }
        
        # Evaluate on test set
        y_pred = self.model_reg.predict(self.X_test)
        y_pred_proba = self.model_reg.predict_proba(self.X_test)
        
        if self.output_size == 1:
            cm = confusion_matrix(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            
            self.reg_metrics = {
                'confusion_matrix': cm,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'fpr': fpr,
                'tpr': tpr,
                'description': reg_description
            }
        else:
            # For multiclass
            y_true = self.y_test
            cm = confusion_matrix(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            
            # Calculate one-vs-rest ROC AUC
            roc_auc = roc_auc_score(self.y_test_onehot, y_pred_proba, multi_class='ovr', average='macro')
            
            self.reg_metrics = {
                'confusion_matrix': cm,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'description': reg_description
            }
        
        # Update plots
        self.update_plots()
        
        self.status_var.set(f"Training with {reg_description} complete")
    
    def update_plots(self):
        # Clear all plots
        self.loss_ax.clear()
        self.accuracy_ax.clear()
        self.cm_ax.clear()
        self.metrics_ax.clear()
        self.roc_ax.clear()
        
        # Plot training and validation loss
        if hasattr(self, 'noreg_results'):
            self.loss_ax.plot(self.noreg_results['train_losses'], label='Train (No Reg)', color='blue')
            self.loss_ax.plot(self.noreg_results['val_losses'], label='Val (No Reg)', color='blue', linestyle='--')
        
        if hasattr(self, 'reg_results'):
            self.loss_ax.plot(self.reg_results['train_losses'], label=f'Train ({self.reg_results["description"]})', color='red')
            self.loss_ax.plot(self.reg_results['val_losses'], label=f'Val ({self.reg_results["description"]})', color='red', linestyle='--')
        
        self.loss_ax.set_title('Loss vs. Epochs')
        self.loss_ax.set_xlabel('Epochs')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.legend()
        self.loss_ax.grid(True)
        self.loss_canvas.draw()
        
        # Plot training and validation accuracy
        if hasattr(self, 'noreg_results'):
            self.accuracy_ax.plot(self.noreg_results['train_accs'], label='Train (No Reg)', color='blue')
            self.accuracy_ax.plot(self.noreg_results['val_accs'], label='Val (No Reg)', color='blue', linestyle='--')
        
        if hasattr(self, 'reg_results'):
            self.accuracy_ax.plot(self.reg_results['train_accs'], label=f'Train ({self.reg_results["description"]})', color='red')
            self.accuracy_ax.plot(self.reg_results['val_accs'], label=f'Val ({self.reg_results["description"]})', color='red', linestyle='--')
        
        self.accuracy_ax.set_title('Accuracy vs. Epochs')
        self.accuracy_ax.set_xlabel('Epochs')
        self.accuracy_ax.set_ylabel('Accuracy')
        self.accuracy_ax.legend()
        self.accuracy_ax.grid(True)
        self.accuracy_canvas.draw()
        
        # Plot confusion matrices
        if hasattr(self, 'noreg_metrics') and hasattr(self, 'reg_metrics'):
            num_plots = 2
            fig = self.cm_fig
            self.cm_ax.clear()
            fig.clear()
            
            for i, (metrics, title) in enumerate([(self.noreg_metrics, 'Without Regularization'), 
                                                 (self.reg_metrics, f'With {self.reg_metrics["description"]}')]):
                ax = fig.add_subplot(1, num_plots, i+1)
                sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f'Confusion Matrix ({title})')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
            
            fig.tight_layout()
            self.cm_canvas.draw()
            
            # Plot comparison of metrics
            fig = self.metrics_fig
            self.metrics_ax.clear()
            fig.clear()
            ax = fig.add_subplot(111)
            
            metrics_names = ['f1_score', 'precision', 'recall', 'roc_auc']
            x = np.arange(len(metrics_names))
            width = 0.35
            
            noreg_values = [self.noreg_metrics[m] for m in metrics_names]
            reg_values = [self.reg_metrics[m] for m in metrics_names]
            
            ax.bar(x - width/2, noreg_values, width, label='Without Regularization')
            ax.bar(x + width/2, reg_values, width, label=f'With {self.reg_metrics["description"]}')
            
            ax.set_title('Performance Metrics Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(['F1 Score', 'Precision', 'Recall', 'ROC AUC'])
            ax.set_ylim([0, 1])
            ax.set_ylabel('Score')
            ax.legend()
            ax.grid(True, axis='y')
            
            fig.tight_layout()
            self.metrics_canvas.draw()
            
            # Plot ROC curves if binary classification
            if self.output_size == 1 and 'fpr' in self.noreg_metrics and 'fpr' in self.reg_metrics:
                fig = self.roc_fig
                self.roc_ax.clear()
                fig.clear()
                ax = fig.add_subplot(111)
                
                ax.plot(self.noreg_metrics['fpr'], self.noreg_metrics['tpr'], 
                        label=f'Without Regularization (AUC = {self.noreg_metrics["roc_auc"]:.3f})')
                ax.plot(self.reg_metrics['fpr'], self.reg_metrics['tpr'], 
                        label=f'With {self.reg_metrics["description"]} (AUC = {self.reg_metrics["roc_auc"]:.3f})')
                ax.plot([0, 1], [0, 1], 'k--')
                ax.set_title('ROC Curve')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend()
                ax.grid(True)
                
                fig.tight_layout()
                self.roc_canvas.draw()
            else:
                # For multiclass, just show the AUC values
                fig = self.roc_fig
                self.roc_ax.clear()
                fig.clear()
                ax = fig.add_subplot(111)
                
                bars = ax.bar([0, 1], [self.noreg_metrics['roc_auc'], self.reg_metrics['roc_auc']], 
                             width=0.6)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.3f}', ha='center', va='bottom')
                
                ax.set_title('ROC AUC (One-vs-Rest)')
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Without Regularization', f'With {self.reg_metrics["description"]}'])
                ax.set_ylim([0, 1])
                ax.set_ylabel('AUC Score')
                ax.grid(True, axis='y')
                
                fig.tight_layout()
                self.roc_canvas.draw()
        elif hasattr(self, 'noreg_metrics'):
            # Only have no regularization metrics
            self.cm_ax.clear()
            sns.heatmap(self.noreg_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=self.cm_ax)
            self.cm_ax.set_title('Confusion Matrix (Without Regularization)')
            self.cm_ax.set_xlabel('Predicted')
            self.cm_ax.set_ylabel('True')
            self.cm_canvas.draw()
            
            # Plot metrics
            self.metrics_ax.clear()
            metrics_names = ['f1_score', 'precision', 'recall', 'roc_auc']
            x = np.arange(len(metrics_names))
            values = [self.noreg_metrics[m] for m in metrics_names]
            
            self.metrics_ax.bar(x, values)
            self.metrics_ax.set_title('Performance Metrics (Without Regularization)')
            self.metrics_ax.set_xticks(x)
            self.metrics_ax.set_xticklabels(['F1 Score', 'Precision', 'Recall', 'ROC AUC'])
            self.metrics_ax.set_ylim([0, 1])
            self.metrics_ax.set_ylabel('Score')
            self.metrics_ax.grid(True, axis='y')
            self.metrics_canvas.draw()
            
            # Plot ROC curve if binary classification
            if self.output_size == 1 and 'fpr' in self.noreg_metrics:
                self.roc_ax.clear()
                self.roc_ax.plot(self.noreg_metrics['fpr'], self.noreg_metrics['tpr'], 
                              label=f'Without Regularization (AUC = {self.noreg_metrics["roc_auc"]:.3f})')
                self.roc_ax.plot([0, 1], [0, 1], 'k--')
                self.roc_ax.set_title('ROC Curve')
                self.roc_ax.set_xlabel('False Positive Rate')
                self.roc_ax.set_ylabel('True Positive Rate')
                self.roc_ax.legend()
                self.roc_ax.grid(True)
                self.roc_canvas.draw()
            else:
                # For multiclass, just show the AUC value
                self.roc_ax.clear()
                self.roc_ax.bar([0], [self.noreg_metrics['roc_auc']], width=0.6)
                self.roc_ax.set_title('ROC AUC (One-vs-Rest)')
                self.roc_ax.set_xticks([0])
                self.roc_ax.set_xticklabels(['Without Regularization'])
                self.roc_ax.set_ylim([0, 1])
                self.roc_ax.text(0, self.noreg_metrics['roc_auc'] + 0.01, 
                              f"{self.noreg_metrics['roc_auc']:.3f}", 
                              ha='center', va='bottom')
                self.roc_ax.set_ylabel('AUC Score')
                self.roc_ax.grid(True, axis='y')
                self.roc_canvas.draw()

# Main function to run the application
def main():
    root = tk.Tk()
    app = NeuralNetworkApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()