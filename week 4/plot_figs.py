from nn_final import generate_classification_data, generate_regression_data
import matplotlib.pyplot as plt 

X, y = generate_regression_data()
plt.plot(X, y)
plt.show()

print(X, y)

X, y = generate_classification_data()
print(X, y)
y = y.reshape(-1)
true = X[y == 1]
false = X[y == 0]
plt.scatter(true[:,0], true[:,1], color='red')
plt.scatter(false[:,0], false[:,1], color='blue')
plt.show()