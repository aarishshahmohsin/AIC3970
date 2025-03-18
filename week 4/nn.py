import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from IPython.display import display, clear_output
import time

np.random.seed(42)

class NeuralNetwork:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, problem_type='classification'):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.problem_type = problem_type
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden1_size) * 0.1
        self.b1 = np.zeros((1, hidden1_size))
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * 0.1
        self.b2 = np.zeros((1, hidden2_size))
        self.W3 = np.random.randn(hidden2_size, output_size) * 0.1
        self.b3 = np.zeros((1, output_size))
        
        # For Adam optimizer
        self.mW1, self.vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        self.mb1, self.vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        self.mW2, self.vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        self.mb2, self.vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        self.mW3, self.vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        self.mb3, self.vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        
        self.param_history = {
            'W1': [], 'b1': [], 'W2': [], 'b2': [], 'W3': [], 'b3': [], 'loss': []
        }
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.power(x, 2)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        
        if self.problem_type == 'classification':
            self.a3 = self.sigmoid(self.z3)
        else:  # regression
            self.a3 = self.z3  # Linear activation for regression
            
        return self.a3
    
    def compute_loss(self, y_pred, y_true):
        if self.problem_type == 'classification':
            epsilon = 1e-15  # Small value to avoid log(0)
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            loss = np.mean((y_pred - y_true) ** 2)
        return loss
    
    def backward(self, X, y, learning_rate=0.01, optimizer='sgd', 
                 beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
        m = X.shape[0]

        # Ensure y is in the correct shape
        y = y.reshape(-1, self.output_size)

        if self.problem_type == 'classification':
            dZ3 = self.a3 - y
        else:  # Regression case: Derivative of MSE loss
            dZ3 = (self.a3 - y) / m  # Derivative of MSE loss

        dW3 = np.dot(self.a2, dZ3.T)
        db3 = np.sum(dZ3, axis=0, keepdims=True)

        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.relu_derivative(self.a2)
        dW2 = np.dot(self.a1.T, dZ2.T)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.a1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update parameters based on optimizer
        if optimizer == 'sgd':
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
            self.W3 -= learning_rate * dW3
            self.b3 -= learning_rate * db3
        elif optimizer == 'momentum':
            # Momentum update
            self.vdW1 = beta1 * self.vdW1 + (1 - beta1) * dW1
            self.vdb1 = beta1 * self.vdb1 + (1 - beta1) * db1
            self.vdW2 = beta1 * self.vdW2 + (1 - beta1) * dW2
            self.vdb2 = beta1 * self.vdb2 + (1 - beta1) * db2
            self.vdW3 = beta1 * self.vdW3 + (1 - beta1) * dW3
            self.vdb3 = beta1 * self.vdb3 + (1 - beta1) * db3
            
            self.W1 -= learning_rate * self.vdW1
            self.b1 -= learning_rate * self.vdb1
            self.W2 -= learning_rate * self.vdW2
            self.b2 -= learning_rate * self.vdb2
            self.W3 -= learning_rate * self.vdW3
            self.b3 -= learning_rate * self.vdb3
        elif optimizer == 'rmsprop':
            # RMSprop update
            self.sdW1 = beta2 * self.sdW1 + (1 - beta2) * (dW1 ** 2)
            self.sdb1 = beta2 * self.sdb1 + (1 - beta2) * (db1 ** 2)
            self.sdW2 = beta2 * self.sdW2 + (1 - beta2) * (dW2 ** 2)
            self.sdb2 = beta2 * self.sdb2 + (1 - beta2) * (db2 ** 2)
            self.sdW3 = beta2 * self.sdW3 + (1 - beta2) * (dW3 ** 2)
            self.sdb3 = beta2 * self.sdb3 + (1 - beta2) * (db3 ** 2)
            
            self.W1 -= learning_rate * dW1 / (np.sqrt(self.sdW1) + epsilon)
            self.b1 -= learning_rate * db1 / (np.sqrt(self.sdb1) + epsilon)
            self.W2 -= learning_rate * dW2 / (np.sqrt(self.sdW2) + epsilon)
            self.b2 -= learning_rate * db2 / (np.sqrt(self.sdb2) + epsilon)
            self.W3 -= learning_rate * dW3 / (np.sqrt(self.sdW3) + epsilon)
            self.b3 -= learning_rate * db3 / (np.sqrt(self.sdb3) + epsilon)
        elif optimizer == 'adam':
            # Adam update
            self.vdW1 = beta1 * self.vdW1 + (1 - beta1) * dW1
            self.vdb1 = beta1 * self.vdb1 + (1 - beta1) * db1
            self.vdW2 = beta1 * self.vdW2 + (1 - beta1) * dW2
            self.vdb2 = beta1 * self.vdb2 + (1 - beta1) * db2
            self.vdW3 = beta1 * self.vdW3 + (1 - beta1) * dW3
            self.vdb3 = beta1 * self.vdb3 + (1 - beta1) * db3
            
            self.sdW1 = beta2 * self.sdW1 + (1 - beta2) * (dW1 ** 2)
            self.sdb1 = beta2 * self.sdb1 + (1 - beta2) * (db1 ** 2)
            self.sdW2 = beta2 * self.sdW2 + (1 - beta2) * (dW2 ** 2)
            self.sdb2 = beta2 * self.sdb2 + (1 - beta2) * (db2 ** 2)
            self.sdW3 = beta2 * self.sdW3 + (1 - beta2) * (dW3 ** 2)
            self.sdb3 = beta2 * self.sdb3 + (1 - beta2) * (db3 ** 2)
            
            # Bias correction
            vdW1_corrected = self.vdW1 / (1 - beta1 ** t)
            vdb1_corrected = self.vdb1 / (1 - beta1 ** t)
            vdW2_corrected = self.vdW2 / (1 - beta1 ** t)
            vdb2_corrected = self.vdb2 / (1 - beta1 ** t)
            vdW3_corrected = self.vdW3 / (1 - beta1 ** t)
            vdb3_corrected = self.vdb3 / (1 - beta1 ** t)
            
            sdW1_corrected = self.sdW1 / (1 - beta2 ** t)
            sdb1_corrected = self.sdb1 / (1 - beta2 ** t)
            sdW2_corrected = self.sdW2 / (1 - beta2 ** t)
            sdb2_corrected = self.sdb2 / (1 - beta2 ** t)
            sdW3_corrected = self.sdW3 / (1 - beta2 ** t)
            sdb3_corrected = self.sdb3 / (1 - beta2 ** t)
            
            self.W1 -= learning_rate * vdW1_corrected / (np.sqrt(sdW1_corrected) + epsilon)
            self.b1 -= learning_rate * vdb1_corrected / (np.sqrt(sdb1_corrected) + epsilon)
            self.W2 -= learning_rate * vdW2_corrected / (np.sqrt(sdW2_corrected) + epsilon)
            self.b2 -= learning_rate * vdb2_corrected / (np.sqrt(sdb2_corrected) + epsilon)
            self.W3 -= learning_rate * vdW3_corrected / (np.sqrt(sdW3_corrected) + epsilon)
            self.b3 -= learning_rate * vdb3_corrected / (np.sqrt(sdb3_corrected) + epsilon)
        elif optimizer == 'adagrad':
            # Adagrad update
            self.cache_W1 += dW1 ** 2
            self.cache_b1 += db1 ** 2
            self.cache_W2 += dW2 ** 2
            self.cache_b2 += db2 ** 2
            self.cache_W3 += dW3 ** 2
            self.cache_b3 += db3 ** 2
            
            self.W1 -= learning_rate * dW1 / (np.sqrt(self.cache_W1) + epsilon)
            self.b1 -= learning_rate * db1 / (np.sqrt(self.cache_b1) + epsilon)
            self.W2 -= learning_rate * dW2 / (np.sqrt(self.cache_W2) + epsilon)
            self.b2 -= learning_rate * db2 / (np.sqrt(self.cache_b2) + epsilon)
            self.W3 -= learning_rate * dW3 / (np.sqrt(self.cache_W3) + epsilon)
            self.b3 -= learning_rate * db3 / (np.sqrt(self.cache_b3) + epsilon)
    
    def train(self, X, y, epochs=100, batch_size=None, learning_rate=0.01, 
              optimizer='sgd', beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Train the neural network.
        """
        m = X.shape[0]
        if batch_size is None:
            batch_size = m
        
        # Reset parameter history
        self.param_history = {
            'W1': [], 'b1': [], 'W2': [], 'b2': [], 'W3': [], 'b3': [], 'loss': []
        }
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Perform forward and backward pass
                y_pred = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate, optimizer, beta1, beta2, epsilon, epoch+1)
                
            # Compute loss for the entire dataset
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            
            # Store current parameters and loss
            self.param_history['W1'].append(self.W1.copy())
            self.param_history['b1'].append(self.b1.copy())
            self.param_history['W2'].append(self.W2.copy())
            self.param_history['b2'].append(self.b2.copy())
            self.param_history['W3'].append(self.W3.copy())
            self.param_history['b3'].append(self.b3.copy())
            self.param_history['loss'].append(loss)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
                
        return self.param_history
    
    def predict(self, X):
        """
        Make predictions for input X.
        """
        return self.forward(X)


def generate_classification_data(n_samples=100, noise=0.1):
    """Generate a simple binary classification dataset."""
    X = np.random.randn(n_samples, 2) * 1.5
    # Create two circular clusters
    y = np.zeros((n_samples, 1))
    for i in range(n_samples):
        if np.sqrt(X[i, 0]**2 + X[i, 1]**2) > 2:
            y[i] = 1
        else:
            y[i] = 0
    # Add some noise
    X += np.random.randn(n_samples, 2) * noise
    return X, y

def generate_regression_data(n_samples=100, noise=0.3):
    """Generate a simple regression dataset."""
    X = np.random.rand(n_samples, 2) * 4 - 2  # Values between -2 and 2
    y = np.sin(X[:, 0]) * np.cos(X[:, 1]).reshape(-1, 1) + np.random.randn(n_samples, 1) * noise
    return X, y

def plot_training_curves(param_history, problem_type, optimizer):
    """Plot the training curves for weights, biases, and loss."""
    epochs = len(param_history['loss'])
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle(f'Training Curves for {problem_type.capitalize()} Problem using {optimizer.capitalize()}', fontsize=16)
    
    # Plot loss
    axes[0, 0].plot(range(epochs), param_history['loss'])
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss Value')
    
    # Plot W1 (only a subset if dimensions are large)
    w1_values = np.array([w[0, 0] for w in param_history['W1']])
    axes[0, 1].plot(range(epochs), w1_values)
    axes[0, 1].set_title('W1[0,0] Value')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Weight Value')
    
    # Plot W2
    w2_values = np.array([w[0, 0] for w in param_history['W2']])
    axes[1, 0].plot(range(epochs), w2_values)
    axes[1, 0].set_title('W2[0,0] Value')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Weight Value')
    
    # Plot W3
    w3_values = np.array([w[0, 0] for w in param_history['W3']])
    axes[1, 1].plot(range(epochs), w3_values)
    axes[1, 1].set_title('W3[0,0] Value')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Weight Value')
    
    # Plot b1
    b1_values = np.array([b[0, 0] for b in param_history['b1']])
    axes[2, 0].plot(range(epochs), b1_values)
    axes[2, 0].set_title('b1[0,0] Value')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Bias Value')
    
    # Plot b2
    b2_values = np.array([b[0, 0] for b in param_history['b2']])
    axes[2, 1].plot(range(epochs), b2_values)
    axes[2, 1].set_title('b2[0,0] Value')
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Bias Value')
    
    # Plot b3
    b3_values = np.array([b[0, 0] for b in param_history['b3']])
    axes[3, 0].plot(range(epochs), b3_values)
    axes[3, 0].set_title('b3[0,0] Value')
    axes[3, 0].set_xlabel('Epoch')
    axes[3, 0].set_ylabel('Bias Value')
    
    # Plot an empty graph or another parameter
    axes[3, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{problem_type}_{optimizer}_training_curves.png')
    plt.show()

def visualize_error_surface(param_history, param_name, fig_title, param_indices=(0, 0)):
    """
    Visualize the error surface with respect to a parameter.
    
    Parameters:
    param_history -- dictionary containing parameter history
    param_name -- name of the parameter to visualize ('W1', 'b1', etc.)
    fig_title -- title for the figure
    param_indices -- indices to extract from the parameter array (for weights only)
    """
    epochs = len(param_history['loss'])
    
    if param_name.startswith('W'):
        # For weight parameters
        i, j = param_indices
        param_values = np.array([p[i, j] for p in param_history[param_name]])
    else:
        # For bias parameters
        i = param_indices[0]
        param_values = np.array([p[0, i] for p in param_history[param_name]])
    
    loss_values = np.array(param_history['loss'])
    
    # Create a figure
    fig = plt.figure(figsize=(12, 6))
    
    # 2D plot
    ax1 = fig.add_subplot(121)
    ax1.plot(param_values, loss_values)
    ax1.set_xlabel(f'{param_name} Value')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'2D Error Surface for {param_name}')
    
    # 3D plot (adding a dimension for visualization)
    ax2 = fig.add_subplot(122, projection='3d')
    
    # For 3D visualization, we'll use the parameter values and epochs as x and y
    x = param_values
    y = np.arange(epochs)
    X, Y = np.meshgrid(x, y)
    Z = np.tile(loss_values, (len(x), 1)).T
    
    # Plot the surface
    surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, antialiased=False, alpha=0.8)
    
    # Add labels
    ax2.set_xlabel(f'{param_name} Value')
    ax2.set_ylabel('Epoch')
    ax2.set_zlabel('Loss')
    ax2.set_title(f'3D Error Surface for {param_name}')
    
    fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)
    fig.suptitle(fig_title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{fig_title.replace(" ", "_")}.png')
    plt.show()

def display_parameter_table(param_history, epochs_to_show=5):
    """Display a table of parameter values for specific epochs."""
    epochs = len(param_history['loss'])
    
    # Select epochs to display (first, last, and some in between)
    if epochs <= epochs_to_show:
        selected_epochs = list(range(epochs))
    else:
        step = max(1, epochs // (epochs_to_show - 1))
        selected_epochs = list(range(0, epochs, step))
        if selected_epochs[-1] != epochs - 1:
            selected_epochs.append(epochs - 1)
    
    # Create a table to display parameter values
    data = []
    headers = ["Epoch", "Loss", "W1[0,0]", "W2[0,0]", "W3[0,0]", "b1[0]", "b2[0]", "b3[0]"]
    
    for epoch in selected_epochs:
        row = [
            epoch, 
            f"{param_history['loss'][epoch]:.6f}",
            f"{param_history['W1'][epoch][0, 0]:.6f}",
            f"{param_history['W2'][epoch][0, 0]:.6f}",
            f"{param_history['W3'][epoch][0, 0]:.6f}",
            f"{param_history['b1'][epoch][0, 0]:.6f}",
            f"{param_history['b2'][epoch][0, 0]:.6f}",
            f"{param_history['b3'][epoch][0, 0]:.6f}"
        ]
        data.append(row)
    
    # Print the table
    print("Parameter Values During Training:")
    print("-" * 80)
    print(f"{headers[0]:^8} {headers[1]:^12} {headers[2]:^12} {headers[3]:^12} {headers[4]:^12} {headers[5]:^12} {headers[6]:^12} {headers[7]:^12}")
    print("-" * 80)
    
    for row in data:
        print(f"{row[0]:^8} {row[1]:^12} {row[2]:^12} {row[3]:^12} {row[4]:^12} {row[5]:^12} {row[6]:^12} {row[7]:^12}")

def train_and_visualize(problem_type, optimizer, epochs=100, learning_rate=0.01, batch_size=1):
    """Train the neural network and visualize the results."""
    if problem_type == 'classification':
        X, y = generate_classification_data(n_samples=200)
        nn = NeuralNetwork(2, 3, 1, 1, problem_type='classification')
    else:  # regression
        X, y = generate_regression_data(n_samples=200)
        nn = NeuralNetwork(2, 3, 1, 1, problem_type='regression')
    
    print(f"\n{'='*50}")
    print(f"Training {problem_type.capitalize()} Model with {optimizer.upper()}")
    print(f"{'='*50}")
    
    # Train the model
    param_history = nn.train(X, y, epochs=epochs, batch_size=batch_size, 
                           learning_rate=learning_rate, optimizer=optimizer)
    
    # Display parameter table
    display_parameter_table(param_history)
    
    # Plot training curves
    plot_training_curves(param_history, problem_type, optimizer)
    
    # Visualize error surfaces for selected parameters
    visualize_error_surface(param_history, 'W1', f'{problem_type.capitalize()} Error Surface for W1 ({optimizer})', (0, 0))
    visualize_error_surface(param_history, 'W2', f'{problem_type.capitalize()} Error Surface for W2 ({optimizer})', (0, 0))
    visualize_error_surface(param_history, 'W3', f'{problem_type.capitalize()} Error Surface for W3 ({optimizer})', (0, 0))
    visualize_error_surface(param_history, 'b1', f'{problem_type.capitalize()} Error Surface for b1 ({optimizer})', (0,))
    visualize_error_surface(param_history, 'b2', f'{problem_type.capitalize()} Error Surface for b2 ({optimizer})', (0,))
    visualize_error_surface(param_history, 'b3', f'{problem_type.capitalize()} Error Surface for b3 ({optimizer})', (0,))
    
    # Visualize the loss landscape
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), param_history['loss'])
    plt.title(f'{problem_type.capitalize()} Loss Curve with {optimizer.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(f'{problem_type}_{optimizer}_loss.png')
    plt.show()
    
    return nn, param_history

# nn, param_history = train_and_visualize(problem_type, optimizer, epochs, learning_rate, batch_size)