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

        self.W1 = np.random.randn(hidden_size, input_size) * 0.01 
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01 
        self.b2 = np.zeros((output_size, 1))

        self.vdW1, self.vdb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.vdW2, self.vdb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.sdW1, self.sdb1 = np.zeros_like(self.W1), np.zeros_like(self.b1)
        self.sdW2, self.sdb2 = np.zeros_like(self.W2), np.zeros_like(self.b2)
        self.t = 0  # time step for Adam 

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
        
        self.update_parameters(dW1, db1, dW2, db2)
        self.history.append((self.W1[0, 0], self.W1[1, 0], self.compute_loss(self.forward(X), Y)))

 

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
            self.W1 -= self.lr * dW1 / (np.sqrt(self.sdW1) + self.epsilon)
            self.b1 -= self.lr * db1 / (np.sqrt(self.sdb1) + self.epsilon)
            self.W2 -= self.lr * dW2 / (np.sqrt(self.sdW2) + self.epsilon)
            self.b2 -= self.lr * db2 / (np.sqrt(self.sdb2) + self.epsilon)
        
        elif self.optimizer == 'Adam':
            self.t += 1
            self.vdW1 = self.beta * self.vdW1 + (1 - self.beta) * dW1
            self.vdb1 = self.beta * self.vdb1 + (1 - self.beta) * db1
            self.sdW1 = self.beta2 * self.sdW1 + (1 - self.beta2) * dW1**2
            self.sdb1 = self.beta2 * self.sdb1 + (1 - self.beta2) * db1**2
            v_corr_dW1 = self.vdW1 / (1 - self.beta**self.t)
            s_corr_dW1 = self.sdW1 / (1 - self.beta2**self.t)
            v_corr_db1 = self.vdb1 / (1 - self.beta**self.t)
            s_corr_db1 = self.sdb1 / (1 - self.beta2**self.t)
            self.W1 -= self.lr * v_corr_dW1 / (np.sqrt(s_corr_dW1) + self.epsilon)
            self.b1 -= self.lr * v_corr_db1 / (np.sqrt(s_corr_db1) + self.epsilon)

        elif self.optimizer == 'Adagrad':
            self.sdW1 += dW1**2
            self.sdb1 += db1**2
            self.W1 -= self.lr * dW1 / (np.sqrt(self.sdW1) + self.epsilon)
            self.b1 -= self.lr * db1 / (np.sqrt(self.sdb1) + self.epsilon)
    

    def train(self, X, Y):
        X = X.T  # Transpose X to match weight matrix multiplication
        Y = Y.reshape(1, -1)  # Ensure Y has shape (1, m)
        
        progress_bar = tqdm(range(self.epochs), desc='Training Progress')
        
        for epoch in progress_bar:
            Y_pred = self.forward(X) 
            loss = self.compute_loss(Y_pred, Y) 
            self.backward(X, Y) 
            
            if epoch % 100 == 0:
                progress_bar.set_postfix(loss=f"{loss:.4f}")  # Updates tqdm bar without breaking it
    

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
            loss_surface[j, i] = nn.compute_loss(Y_pred, Y.reshape(1, -1))  # ✅ Fix indexing

    nn.W1 = original_W1.copy()
    return w1_vals, w2_vals, loss_surface



def animate_loss_surface(w1_vals, w2_vals, loss_surface, optimizer_path):
    W1, W2 = np.meshgrid(w1_vals, w2_vals)
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plot loss surface
    ax.plot_surface(W1, W2, loss_surface, cmap='viridis', alpha=0.8)

    ax.set_xlabel('Weight W1[0,0]')
    ax.set_ylabel('Weight W1[1,0]')
    ax.set_zlabel('Loss')
    ax.set_title('3D Loss Surface with Optimizer Path')

    path = np.array(optimizer_path)

    # Ensure path has the correct shape
    if path.ndim != 2 or path.shape[1] != 3:
        print("Error: Optimizer path must be (N, 3), but got", path.shape)
        return
    
    # Initial plot for optimizer path
    trajectory, = ax.plot([], [], [], marker='o', color='r', markersize=3, label="Optimizer Trajectory")

    def update(frame):
        trajectory.set_data(path[:frame+1, 0], path[:frame+1, 1])
        trajectory.set_3d_properties(path[:frame+1, 2])
        return trajectory,

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=100, blit=True)

    plt.legend()
    plt.show()
    return ani




optimizer_list = [
    'SGD',
    'Momentum',
    'RMSprop',
    'Adam',
    'Adagrad'
]

df = pd.read_csv('week 3/boston_house_prices.csv', skiprows=1)
X = df.drop(columns=['MEDV']).to_numpy()
y = df['MEDV'].to_numpy()

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = (y - np.mean(y)) / np.std(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


for optimizer in optimizer_list:
    print(optimizer)
    nn = NeuralNetwork(input_size=X.shape[1], hidden_size=10, output_size=1, optimizer=optimizer, epochs=10000)
    print(nn.history)
    nn.train(X_train, y_train)
    w1_vals, w2_vals, loss_surface = compute_loss_surface(nn, X, y)
    animate_loss_surface(w1_vals, w2_vals, loss_surface, nn.history)
    
