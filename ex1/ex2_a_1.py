"""
ex1 - 2a1
@author: Noam Mirjani & Yaakov Haimoff 
        315216515        318528520
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.gridspec import GridSpec



def hist(x, scores, x_data, s):
    ''' calc all the data thats needed to the histogram and plot and build them
        Arguments: x = the asex type subplot
                    scored = group of rand data 
                    x_data = the plot x asix
                    s = string name
    '''
    
    mu, sigma = norm.fit(scores)
    y_data = norm.pdf(x_data, loc=mu, scale=sigma)

    if s == "x1":
        x.hist(scores, bins=30, density=True, ec='w', lw=1,zorder=2, fc='royalblue')
        x.plot(x_data,y_data, c='r')
    else:
        x.hist(scores, bins=30, density=True, orientation='horizontal',ec='w', lw=1,zorder=2, fc='royalblue')
        x.plot(y_data, x_data, c='r')


def grid(x):
    ''' design the grid
        Arguments: x = the asex type subplot
    '''
    x.grid(True, alpha=0.5, zorder=1, color='w')
    x.set_facecolor("lavender")
    

# define ths sub plots data
fig = plt.figure(figsize=(9,9))
gs = GridSpec(nrows=2, ncols=2, figure=fig)
ax1 = fig.add_subplot(gs[1,1])
ax2 = fig.add_subplot(gs[0, 0])
ax3 = fig.add_subplot(gs[0,1], sharey=ax2, sharex=ax1)


# Fit and find the normal distribution to the data
X1 = np.random.normal(0, 1, 1000)
X2 = list(map(lambda x: -x if abs(x)>1 else x, X1))


# Plot the PDF.
x_data = np.arange(-5, 5, 0.01)


#Normalised histogram & distribution x1 and x2
hist(ax1, X1, x_data, "x1")
hist(ax2, X2, x_data, "x2")

# common function hidtogram
ax3.scatter(X1, X2,zorder=2, fc='royalblue', lw=1, marker='.')


# Add a grids
grid(ax1)
grid(ax2)
grid(ax3)


plt.show()