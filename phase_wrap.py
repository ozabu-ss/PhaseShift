from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
import numpy as np


# Create phase function.
X = np.arange(-2, 2, 0.05)
Y = np.arange(-2, 2, 0.05)
X, Y = np.meshgrid(X, Y)
R = X*np.exp(-X**2 - Y**2)
Z = np.sin(R) * 50  + np.random.random(X.shape) * 2

def wrap(z):
	while np.abs(z) > np.pi:
		z = z - np.sign(z) * 2*np.pi
	return z

def array_for(z):
	return np.array([wrap(zi) for zi in z])

Z_wrapped = np.array(map(array_for, Z))
delta_x = np.diff(Z_wrapped, axis = 0)
delta_y = np.diff(Z_wrapped, axis = 1)
delta_x = np.array(map(array_for, delta_x))
delta_y = np.array(map(array_for, delta_y))

b = np.zeros(Z.shape)
for i in range(1, Z.shape[0]):
	for j in range(1, Z.shape[1]):
		b[i][j] = delta_x[i-1][j] + delta_y[i][j-1] - 2 * Z_wrapped[i][j] + Z_wrapped[i-1][j] + Z_wrapped[i][j-1]
		b[i][j] = b[i][j] / (2 * np.pi)

#compute k
k = np.zeros(Z.shape)
for i in range(1, Z.shape[0]):
	for j in range(1, Z.shape[1]):
		k[i][j] = (b[i][j] + k[i-1][j] + k[i][j-1])/2

#compute unwrapped Z
Z_unwrapped = Z_wrapped + 2*np.pi*np.round(k)

# Plot the surfaces
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(1, 3, 1, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
levels = MaxNLocator(nbins=15).tick_values(Z_wrapped.min(), Z_wrapped.max())
ax = fig.add_subplot(1, 3, 2)
surf = ax.contourf(X, Y, Z_wrapped, cmap=cm.coolwarm,
                       levels = levels)
ax = fig.add_subplot(1, 3, 3, projection='3d')
surf = ax.plot_surface(X, Y, Z_unwrapped, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()
