# -*- coding: utf-8 -*-
"""
@author: noammirjani 315216515 :)
"""
from create_data import create_data,show_data
from sklearn.neural_network import MLPClassifier

# create the data 
n_samples = 400     # 400 samples for the whole data set
test_percent = 0.5  # half to test and half to train 
scale_factor = 2.5  # scale for display purpose

 
X_train, X_test, y_train, y_test = create_data(
    n_samples, test_percent, scale_factor, 1)     # using 1 as seed for random 


#%% create the NN and train
# multi layer perceptron as classifier
max_iter = 2000
param =  {        
        "solver": "sgd",
        "activation": "tanh",
        "learning_rate": "constant",
        "momentum": 0,
        "alpha": 0,
        "tol": 1e-5,
        "learning_rate_init": 0.3,
    }
mlp = MLPClassifier(hidden_layer_sizes= (4,2),
                    random_state=0,
                    max_iter=max_iter,
                    **param)

mlp.fit(X_train, y_train)
# print the wanted data
show_data(X_train, X_test, y_train, y_test, mlp)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Training set loss: %f" % mlp.loss_)
print("Test set score: %f" % mlp.score(X_test, y_test))


