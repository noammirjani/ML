"""
ex1 - 2b1
@author: Noam Mirjani & Yaakov Haimoff 
        315216515        318528520
"""

# Importing the necessary modules
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm


# Our 2-dimensional distribution will be over variables X and Y
N = 100
axis = np.linspace(-5, 5, N)
X, Y = np.meshgrid(axis, axis)


# Fit and find the normal distribution to the data
x1 = np.random.normal(0, 1, 1000)
x2 = list(map(lambda x: -x if abs(x)>1 else x, x1))


# get the two pdf from x1, x2
pdf1 = norm.pdf(axis, loc=0, scale=1)
pdf2 = norm.pdf(axis, loc=0, scale=1)


# get the Z axis 
pdf1, pdf2 = np.meshgrid(pdf1, pdf2)
Z = pdf1 * pdf2


# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=1, antialiased=True,cmap=cm.viridis)
cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)


# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)
ax.set_facecolor("lavender")


plt.show()