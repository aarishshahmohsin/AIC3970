import numpy as np 
from tqdm import tqdm
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import StandardScaler 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, optimizer='SGD', lr=0.01, beta=0.9, beta2=0.999, epsilon=1e-8, epochs=1000):
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.lr = lr 
        self.beta = beta 
        self.beta2 = beta2 
        self.epsilon = epsilon
        self.epochs = epochs
        self.history = []
        self.loss_history = []  
        self.param_history = []

        self.W1 = np.ones((hidden_size, input_size)) * 1.5
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.ones((output_size, hidden_size)) * 1.5
        self.b2 = np.zeros((output_size, 1))

        self.vdW1, self.vdb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vdW2, self.vdb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.sdW1, self.sdb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.sdW2, self.sdb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.t = 0

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
        m = Y.shape[1]
        return (1/(2*m)) * np.sum((Y_pred - Y) ** 2)

    def backward(self, X, Y):
        m = X.shape[1]
        dZ2 = self.Z2 - Y
        dW2 = (1/m) * np.dot(dZ2, self.A1.T)
        db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = (1/m) * np.dot(dZ1, X.T)
        db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
        
        current_loss = self.compute_loss(self.forward(X), Y)
        self.history.append((self.W1[0, 0], self.W1[1, 0], current_loss))
        self.loss_history.append(current_loss)
        self.update_parameters(dW1, db1, dW2, db2)

    def update_parameters(self, dW1, db1, dW2, db2):
        if self.optimizer == 'SGD':
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

        elif self.optimizer == 'Momentum':
            self.vdW1 = self.beta * self.vdW1 + (1 - self.beta) * dW1
            self.vdb1 = self.beta * self.vdb1 + (1 - self.beta) * db1
            self.vdW2 = self.beta * self.vdW2 + (1 - self.beta) * dW2
            self.vdb2 = self.beta * self.vdb2 + (1 - self.beta) * db2
            
            self.W1 -= self.lr * self.vdW1
            self.b1 -= self.lr * self.vdb1
            self.W2 -= self.lr * self.vdW2
            self.b2 -= self.lr * self.vdb2

        elif self.optimizer == 'RMSprop':
            self.sdW1 = self.beta * self.sdW1 + (1 - self.beta) * dW1**2
            self.sdb1 = self.beta * self.sdb1 + (1 - self.beta) * db1**2
            self.sdW2 = self.beta * self.sdW2 + (1 - self.beta) * dW2**2
            self.sdb2 = self.beta * self.sdb2 + (1 - self.beta) * db2**2
            
            self.W1 -= self.lr * dW1 / (np.sqrt(self.sdW1 + self.epsilon))
            self.b1 -= self.lr * db1 / (np.sqrt(self.sdb1 + self.epsilon))
            self.W2 -= self.lr * dW2 / (np.sqrt(self.sdW2 + self.epsilon))
            self.b2 -= self.lr * db2 / (np.sqrt(self.sdb2 + self.epsilon))

        elif self.optimizer == 'Adam':
            self.t += 1
            
            self.vdW1 = self.beta * self.vdW1 + (1 - self.beta) * dW1
            self.vdb1 = self.beta * self.vdb1 + (1 - self.beta) * db1
            self.vdW2 = self.beta * self.vdW2 + (1 - self.beta) * dW2
            self.vdb2 = self.beta * self.vdb2 + (1 - self.beta) * db2
            
            self.sdW1 = self.beta2 * self.sdW1 + (1 - self.beta2) * dW1**2
            self.sdb1 = self.beta2 * self.sdb1 + (1 - self.beta2) * db1**2
            self.sdW2 = self.beta2 * self.sdW2 + (1 - self.beta2) * dW2**2
            self.sdb2 = self.beta2 * self.sdb2 + (1 - self.beta2) * db2**2
            
            vdW1_corrected = self.vdW1 / (1 - self.beta**self.t)
            vdb1_corrected = self.vdb1 / (1 - self.beta**self.t)
            vdW2_corrected = self.vdW2 / (1 - self.beta**self.t)
            vdb2_corrected = self.vdb2 / (1 - self.beta**self.t)
            
            sdW1_corrected = self.sdW1 / (1 - self.beta2**self.t)
            sdb1_corrected = self.sdb1 / (1 - self.beta2**self.t)
            sdW2_corrected = self.sdW2 / (1 - self.beta2**self.t)
            sdb2_corrected = self.sdb2 / (1 - self.beta2**self.t)
            
            self.W1 -= self.lr * vdW1_corrected / (np.sqrt(sdW1_corrected) + self.epsilon)
            self.b1 -= self.lr * vdb1_corrected / (np.sqrt(sdb1_corrected) + self.epsilon)
            self.W2 -= self.lr * vdW2_corrected / (np.sqrt(sdW2_corrected) + self.epsilon)
            self.b2 -= self.lr * vdb2_corrected / (np.sqrt(sdb2_corrected) + self.epsilon)

        elif self.optimizer == 'Adagrad':
            self.sdW1 += dW1**2
            self.sdb1 += db1**2
            self.sdW2 += dW2**2
            self.sdb2 += db2**2
            
            self.W1 -= self.lr * dW1 / (np.sqrt(self.sdW1 + self.epsilon))
            self.b1 -= self.lr * db1 / (np.sqrt(self.sdb1 + self.epsilon))
            self.W2 -= self.lr * dW2 / (np.sqrt(self.sdW2 + self.epsilon))
            self.b2 -= self.lr * db2 / (np.sqrt(self.sdb2 + self.epsilon))

    def train(self, X, Y):
        X = X.T
        Y = Y.reshape(1, -1)
        
        progress_bar = tqdm(range(self.epochs), desc=f'Training ({self.optimizer})')
        
        for epoch in progress_bar:
            Y_pred = self.forward(X)
            loss = self.compute_loss(Y_pred, Y)
            self.backward(X, Y)
            
            if epoch % 100 == 0:
                progress_bar.set_postfix(loss=f"{loss:.4f}")

def compute_loss_surface(nn, X, Y, w1_range=(-2, 2), w2_range=(-2, 2), resolution=100):
    """Compute the loss surface for visualization"""
    w1_vals = np.linspace(w1_range[0], w1_range[1], resolution)
    w2_vals = np.linspace(w2_range[0], w2_range[1], resolution)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    loss_surface = np.zeros_like(W1)
    
    original_W1 = nn.W1.copy()
    
    X = X.T
    Y = Y.reshape(1, -1)
    
    for i in range(resolution):
        for j in range(resolution):
            nn.W1[0, 0] = W1[i, j]
            nn.W1[1, 0] = W2[i, j]
            Y_pred = nn.forward(X)
            loss_surface[i, j] = nn.compute_loss(Y_pred, Y)
    
    nn.W1 = original_W1
    return W1, W2, loss_surface

def visualize_training(nn, w1_vals, w2_vals, loss_surface):
    """Create static visualization of the training process"""
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131)
    path = np.array(nn.history)
    contour = ax1.contour(w1_vals, w2_vals, loss_surface, levels=50)
    ax1.plot(path[:, 0], path[:, 1], 'r.-', label='Optimization Path')
    ax1.set_title(f'Loss Surface Contour ({nn.optimizer})')
    ax1.set_xlabel('W1[0,0]')
    ax1.set_ylabel('W1[1,0]')
    plt.colorbar(contour, ax=ax1)
    
    ax2 = fig.add_subplot(132, projection='3d')
    surf = ax2.plot_surface(w1_vals, w2_vals, loss_surface, cmap='viridis', alpha=0.8)
    ax2.plot(path[:, 0], path[:, 1], path[:, 2], 'r.-', label='Optimization Path')
    ax2.set_title('3D Loss Surface')
    ax2.set_xlabel('W1[0,0]')
    ax2.set_ylabel('W1[1,0]')
    ax2.set_zlabel('Loss')
    
    print(nn.loss_history[0])
    ax3 = fig.add_subplot(133)
    ax3.plot(nn.loss_history, 'b-')
    ax3.set_title('Loss History')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

optimizer_list = ['SGD', 'Momentum', 'RMSprop', 'Adam', 'Adagrad']

df = pd.read_csv('/Users/aarish/AIC3970/week 3/boston_house_prices.csv', skiprows=1)
X = df['CRIM'].to_numpy()
y = df['MEDV'].to_numpy()

# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# y = (y - np.mean(y)) / np.std(y)
X = X.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for optimizer in optimizer_list:
    nn = NeuralNetwork(
        input_size=X.shape[1], 
        hidden_size=10, 
        output_size=1, 
        optimizer=optimizer, 
        epochs=1000,
        lr=0.01 if optimizer in ['SGD', 'Momentum'] else 0.001
    )
    
    nn.train(X_train, y_train)
    
    w1_vals, w2_vals, loss_surface = compute_loss_surface(nn, X_train, y_train)
    
    # Visualize results
    visualize_training(nn, w1_vals, w2_vals, loss_surface)