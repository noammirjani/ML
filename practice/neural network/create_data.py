# -*- coding: utf-8 -*-
"""
@author: noammirjani 315216515 :)
"""
# import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay


# create data -> parameters: parameters for the data split
#               returm:  X_train, X_test, y_train, y_test
def create_data(n_samples = 400, test_percent = 0.5,
                scale_factor = 1, random_state = None):
    
    X,y = make_circles(n_samples=n_samples, noise=0.1,
                       random_state=random_state, factor=0.35 )
    
    X = StandardScaler().fit_transform(X)
    X = X * scale_factor
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_percent, random_state=42)
    
    return X_train, X_test, y_train, y_test



# show_data  -> parameters: parameters the groups data and the classifier
#               returm:  void 
# plots the training loss curve & the prediction results
def show_data(X_train, X_test, y_train, y_test, mlp):            
    # figure with 2 subplots
    fig, axs = plt.subplots(2, 2,figsize=(10,10))
    fig.set_facecolor("grey")
    
    # define colormap with 2 colors: orange and blue
    cm_bright = ListedColormap(["#FF8800", "white", "#0000FF"])
    # define colormap with 2 colors: light orange and light blue for the classification
    #  presentaion (better colors then using the alpha parameter in DecisionBoundaryDisplay)
    cm_light = ListedColormap(["#F5BC5F","white", "#8FADF9"])
    
    # subplots titles
    titles = ['Prediction space', 'Prediction + training data', 
          'Prediction + test data', 'Prediction + training + test data']
    
    # setting axis limits according to the data
    _min = round(min(np.min(X_train),np.min(X_test)))
    _max = round(max(np.max(X_train),np.max(X_test)))
                          
    y_min = x_min = _min
    y_max = x_max = _max 
    
    # Loop over the subplots and set the title and plot data
    for i, ax in enumerate(axs.flatten()):
        # color the prediction areas
        DecisionBoundaryDisplay.from_estimator(mlp, X_test, cmap=cm_light, ax=ax, eps=1.1)
        
        #desplay the function
        if i == 1 or i == 3:
            ax.scatter(X_train[:,0], X_train[:, 1], c=y_train,cmap=cm_bright, ec="w", lw=2)
        
        if i == 2 or i == 3:
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, ec="k")
        #title and asixs fix  
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', 'box')
        ax.set_title(titles[i])

    # define new figure and print the loss function
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot()
    ax.plot(mlp.loss_curve_)
    ax.set_xlabel("iteration #")
    ax.set_ylabel("loss")
    ax.set_title("Training loss curve")
    
    
    plt.show()


if __name__ == "__main__":
    
    n_samples = 400     # 400 samples for the whole data set
    test_percent = 0.5  # half to test and half to train 
    scale_factor = 2.4  # scale for display purpose
    random_state = 1    # to have the same data 
    
    X_train, X_test, y_train, y_test = create_data(
        n_samples, test_percent, scale_factor, random_state)    
    
   # show_data(X_train, X_test, y_train, y_test)
    