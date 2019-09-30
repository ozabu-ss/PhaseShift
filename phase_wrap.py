from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # <-- required by subplot projection
import numpy as np


def calculate_something(X, Y):
    # Create phase function.
    R = X * np.exp(-X ** 2 - Y ** 2)
    Z = np.sin(R) * 50 + np.random.random(X.shape) * 2

    wrap = np.vectorize(lambda z: (z + np.pi) % (2 * np.pi) - np.pi)

    Z_wrapped = wrap(Z)
    shift_x, shift_y = np.roll(Z_wrapped, 1, axis=0), np.roll(Z_wrapped, 1, axis=1)
    delta_x, delta_y = wrap(Z_wrapped - shift_x), wrap(Z_wrapped - shift_y)

    b = (delta_x + delta_y + shift_x + shift_y - 2 * Z_wrapped) / (2 * np.pi)

    k = np.zeros(Z.shape)
    for i in range(1, Z.shape[0]):
        for j in range(1, Z.shape[1]):
            k[i][j] = (b[i][j] + k[i - 1][j] + k[i][j - 1]) / 2

    Z_unwrapped = Z_wrapped + 2 * np.pi * np.round(k)

    return Z, Z_unwrapped, Z_wrapped


if __name__ == '__main__':
    X, Y = np.meshgrid(np.arange(-2, 2, 0.05), np.arange(-2, 2, 0.05))
    Z, Z_unwrapped, Z_wrapped = calculate_something(X, Y)

    # Plot the surfaces
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    levels = MaxNLocator(nbins=15).tick_values(Z_wrapped.min(), Z_wrapped.max())
    ax = fig.add_subplot(1, 3, 2)
    surf = ax.contourf(X, Y, Z_wrapped, cmap=cm.coolwarm, levels=levels)
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    surf = ax.plot_surface(X, Y, Z_unwrapped, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()
