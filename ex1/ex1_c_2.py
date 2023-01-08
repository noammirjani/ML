"""
ex1 - 1c2
@author: Noam Mirjani & Yaakov Haimoff 
        315216515        318528520
"""


import matplotlib.pyplot as plt
import numpy as np


# data
signal = [0,0,0,0,1,1,1,1,0,0,0,0]
kernel = [0, 1, -2, 1, 0]
convolution = np.convolve(signal,kernel, 'same')


# define ths sub plots data
fig, axs = plt.subplots(3, sharex=True, sharey=True)
fig.suptitle('ex1 c2')           


#signal
axs[0].set_title('signal', y=0.6,fontsize='large')
axs[0].step(np.arange(0,12), signal, color='b', where='mid')


#kernel
axs[1].set_title('kernel',  y=0.6, fontsize='large')
axs[1].step(np.arange(0,5), kernel, color='g',where='mid')


#result
axs[2].set_title('result',  y=0.7, fontsize='large')
axs[2].step(np.arange(0,12),convolution, color='r', where='mid')


plt.show()