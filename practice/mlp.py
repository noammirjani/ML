import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


# FIGURE AND ASIX
fig, axs = plt.subplots(1,1, figsize=(8,8))

 # preprocess dataset, split into training and test part
X,Y =  make_circles(n_samples=400, noise=0.08, factor=0.3, random_state=1)
X = X*[5,5]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.5, train_size=0.5)


# Define classifier and train the sets 
mlp = MLPClassifier(hidden_layer_sizes=(4, 2), max_iter=1500)
mlp.fit(X_train,y_train)


# Using the trained network to predict
predict = mlp.predict(X_test)


# Print data
accuracy = accuracy_score(y_test,predict)*100
confusion_mat = confusion_matrix(y_test,predict)

print("Accuracy for Neural Network is:",accuracy, "\n------\n")
print("Confusion Matrix\n", confusion_mat, "\n------\n")
print("report\n", classification_report(y_test,predict),"\n------\n")
