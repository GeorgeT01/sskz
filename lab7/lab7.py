import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm




data_row1 = np.genfromtxt('ex1data1.txt', delimiter=',')
data1 = pd.DataFrame(data_row1, columns=list(['Population', 'Profit']))
data1.head()


plt.figure()
data1.plot(x='Population', y='Profit',  style=['b.'])
plt.grid(True)
plt.show()



def compute_cost(X, y, theta):
    h = [np.matmul(x, theta.T).sum() for x in X]
    return np.power(h - y, 2).sum() / (2 * m)



m = data_row1.shape[0] # Size of training set
n = data_row1.shape[1] # Size of feature vector
X1 = data1[['Population']]
X1.insert(0, 'theta_0', 1)
y1 = data1['Profit']
# theta the coefficients of the linear equation
theta = np.zeros((1, n))
X1.head()



cost = compute_cost(X1.to_numpy(), y1.to_numpy(), theta)
print('Cost =', cost)



'''
function [theta, J_history] = gradient_dscent(X, y, theta, alpha, number_iterations)
taking num_iters gradient steps with learning rate alpha
'''
def gradient_descent(X, y, theta, alpha, number_iterations):
    m = y.shape[0] 
    n = X.shape[1]
    j_history = []
    for i in range(0, number_iterations):
        deltas = np.zeros(n)
        for j in range(0, n):
            xj = X[:, j]
            h = [np.matmul(x, theta.T)[0] for x in X]
            deltas[j] = ((h - y) * xj).sum() * alpha / m
        theta[0] -= deltas
        j_history.append(compute_cost(X, y, theta))

    return theta, j_history

iterations = 1500
alpha = 0.01
(theta, j_history) = gradient_descent(X1.to_numpy(), y1.to_numpy(), theta, alpha, iterations)

print('gradient descent thera: ', theta)

print('for population = 35,000, profit prediction: ', (np.matmul([1, 3.5], theta.T).sum() * 10000)) #1st predict
print('for population = 70,000, profit prediction', (np.matmul([1, 7], theta.T).sum() * 10000)) #2nd predict


h = [np.matmul(x, theta.T).sum() for x in X1.to_numpy()]
data1_plot = data1.join(pd.DataFrame({'Linear regression': h}))


plt.figure()
ax = data1_plot.plot(x='Population', y='Profit',  style=['b.'])
data1_plot.plot(x='Population', y='Linear regression', ax=ax)
plt.grid(True)
plt.show()


theta0_vals = np.linspace(-10, 10, num=100)
theta1_vals = np.linspace(-1, 4, num=100)
# initialize J values to a matrix of 0
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))



# Fill J values
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([[theta0_vals[i], theta1_vals[j]]])
        J_vals[i, j] = compute_cost(X1.to_numpy(), y1.to_numpy(), t)
        
J_vals = J_vals.T


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(theta0_vals,theta1_vals,J_vals,cmap="coolwarm")
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'J($\theta$)')
ax.view_init(30,120)#rotate for better angle
plt.show()




# Contour plot
plt.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax = plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
#plt.clabel(ax, inline=0, fontsize=2)
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')
plt.plot(theta[0, 0], theta[0, 1], 'rx', linewidth=1, markersize=15)
plt.show()