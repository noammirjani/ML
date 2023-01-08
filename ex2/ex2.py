"""
noam mirjani    315216515
 yaakov haimoff 318528510

"""

import numpy as np
from sklearn import datasets, metrics, svm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

def misclassification():
    ''' prints all the misclassification nambers from the data set,
        above the number is the true label and next to it the wrong predicted'''
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)
    
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False)

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    #define figure data
    fig = plt.figure(figsize=(10,4))
    fig.patch.set_facecolor('xkcd:gray')
    fig.suptitle('Test. mis-classification: expected - predicted', fontsize=16)

    index=0
    for image, prediction, trueLabel in zip(X_test, predicted, y_test):
        if prediction != trueLabel:
            ax = fig.add_subplot(3, 10, index+1)
            ax.set_axis_off()
            image = image.reshape(8, 8)
            ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            ax.set_title(f"{trueLabel} {prediction}")
            index += 1
            

''' SECTION 20-B '''
#               FEATURE 1  - SUM MATRIX
def get_mat_sum(matrix):
    ''' :param matrix: matrix of image(8x8) 
        :return: the sum of values in the matrix'''
    return matrix.sum()/8


#               FEATURE 2  - VARIANCE OF ROWS SUM
def get_rows_sum_var(matrix):
    ''' :param matrix: matrix of image(8x8) 
        :return: the variance of the rows sum'''
    return np.var(np.sum(matrix, axis=1))


#               FEATURE 3  - VARIANCE OF COLS SUM
def get_cols_sum_var(matrix):
    ''' :param matrix: matrix of image(8x8) 
        :return: variance of the cols' sum'''
    return np.var(np.sum(matrix, axis=0))


#              FEATURE 4 - SUM OF MAIN AREA IN MATRIX
def get_main_area_sum(matrix):
    '''# The images are in six[8,8], the main aria found in the center, [4,4] matrix.
        :param matrix: matrix of image(8x8) 
        :return: sum of the main area in the matrix'''
    temp_sum = 0
    for i in range(2, 6):
        for j in range(2, 6):
            temp_sum += matrix[i][j]
    return temp_sum


#              FEATURE 5 - MEASURE OF VERTICALLY SYMMETRIC MATRIX
def get_measure_vertically_symmetric(matrix):
    ''' Compute the difference between two cells which are parallel, and we sum the
        :param matrix: matrix of image(8x8) 
        :return: the measure of vertical Symmetry'''
    n = len(matrix)
    measure = 0
    for i in range(len(matrix[0])):
        for j in range(n // 2):
            measure += abs(matrix[j][i] - matrix[n - 1 - j][i])
    return measure


#              FEATURE 6 - MEASURE OF HORIZONTALLY SYMMETRIC MATRIX
def get_measure_horizontally_symmetric(matrix):
    ''' Compute the difference between two cells which are parallel, and we sum the
        :param matrix: matrix of image(8x8) 
        :return: the measure of horizontally Symmetry'''
    n = len(matrix[0])
    measure = 0
    for i in range(len(matrix)):
        for j in range(n // 2):
            measure += abs(matrix[i][j] - matrix[i][n - 1 - j])
    return measure
  
 
def run_feature(digits_images, feature_function):
    ''' :param digits_images: array of matrixes of images(8x8),  
               feature_function: the wanted feature function that gets a matrix and returns number
        :return: array that contains the features data set'''
    feature_array = []
    for image_matrix in digits_images:
        feature_array.append(feature_function(image_matrix))
    return feature_array


def scatter_one_feature():
    ''' function generates a scatter plot for one feature of the data, with the x-axis
        representing the feature and the y-axis representing the "digit"'''
    # loop through the features
    for index in range(0, len(features_list), 2):
        # add the new feature to the data array
        new_data = np.column_stack((data, features_list[index]))
        
        # create a new figure
        fig = plt.figure()
        ax = plt.subplot()
        # generate a scatter plot with the new data
        ax.scatter(new_data[indices_0_1, -1], selected_data, c=selected_data)
        
        # Add labels to the axes
        plt.xlabel(f'{features_list[index+1]}')
        plt.ylabel('digit')
        fig.suptitle(f"feature: {features_list[index+1]}, for '0' and '1' digits", fontsize=10)
        plt.tight_layout()
        plt.show()


def scatter_two_features():
    '''
    generates a scatter plot for two features of the data, with the x-axis 
    representing the first feature, the y-axis representing the second feature, 
    and the "digit" represented by the color of the points.
    '''
    # Loop through the items and create pairs
    for i in range(0, len(features_list), 2):
        for j in range(i + 2, len(features_list), 2):
            # add the sums as a new feature to the data array
            data1 = np.column_stack((data, features_list[i]))
            data2 = np.column_stack((data, features_list[j]))

            # create figure plot
            figure = plt.figure()
            ax = plt.subplot()
            ax.scatter(data1[indices_0_1, -1], data2[indices_0_1, -1], c=selected_data)

            # Add labels to the axes
            plt.xlabel(f'{features_list[i + 1]}')
            plt.ylabel(f'{features_list[j + 1]}')
            figure.suptitle(f"feature: {features_list[i + 1]} and {features_list[j + 1]} " 
                            f"of matrix, for '0' and '1' digits", fontsize=10)
            plt.tight_layout()
            plt.show()
        
  
def scatter_three_features():
    """generates a 3D scatter plot for three features of the data, with the x-axis, 
       y-axis,and z-axis representing the three features and the "digit" represented
       by the color of the points."""
    # Loop through the items and create pairs
    for i in range(0, len(features_list), 2):
        for j in range(i + 2, len(features_list), 2):
            for k in range(j + 2, len(features_list), 2):
                # add the sums as a new feature to the data array
                data1 = np.column_stack((data, features_list[i]))
                data2 = np.column_stack((data, features_list[j]))
                data3 = np.column_stack((data, features_list[k]))

                # create figure plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(data1[indices_0_1, -1], data2[indices_0_1, -1], data3[indices_0_1, -1], c=selected_data)

                # Add labels to the axes
                ax.set_xlabel(f'{features_list[i + 1]}')
                ax.set_ylabel(f'{features_list[j + 1]}')
                ax.set_zlabel(f'{features_list[k + 1]}')
                fig.suptitle(f"feature: {features_list[i + 1]} and {features_list[j + 1]} " 
                             f"and {features_list[k + 1]} of matrix, for '0' and '1' digits", fontsize=10)
                plt.tight_layout()
                plt.show()
             
                
def logistic_regression_clf():
    ''' computes logistic regression clf for two features'''
    # Loop through the items and create pairs
    for i in range(0, len(features_list), 2):
        for j in range(i + 2, len(features_list), 2):
            # creating the X (feature) matrix
            x = np.column_stack((features_list[i], features_list[j]))
            # scaling the values for better classification performance
            x_scaled = preprocessing.scale(x[indices_0_1])
            # the predicted outputs
            y = digits.target[indices_0_1]
            # Training Logistic regression
            logistic_classifier = LogisticRegression(solver='lbfgs')
            logistic_classifier.fit(x_scaled, y)
            # show how good is the classifier on the training data
            expected = y
            predicted = logistic_classifier.predict(x_scaled)
            print(f"Logistic regression using [{features_list[i+1]}, {features_list[j+1]}] features:\n%s\n" % (
            metrics.classification_report(expected, predicted)))
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
            # estimate the generalization performance using cross validation
            predicted2 = cross_val_predict(logistic_classifier, x_scaled, y, cv=10)
            print(f"Logistic regression using {features_list[i+1]}, {features_list[j+1]} features cross validation:\n%s\n" % (
            metrics.classification_report(expected, predicted2)))
            print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))


def logistic_regression_clf_for_three():
    ''' computes logistic regression clf for three features'''
    # Loop through the items and create pairs
    for i in range(0, len(features_list), 2):
        for j in range(i + 2, len(features_list), 2):
            for k in range(j + 2, len(features_list), 2):
                # creating the X (feature) matrix
                x = np.column_stack((features_list[i], features_list[j], features_list[k]))
                # scaling the values for better classification performance
                x_scaled = preprocessing.scale(x[indices_0_1])
                # the predicted outputs
                y = digits.target[indices_0_1]
                # Training Logistic regression
                logistic_classifier = LogisticRegression(solver='lbfgs')
                logistic_classifier.fit(x_scaled, y)
                # show how good is the classifier on the training data
                expected = y
                predicted = logistic_classifier.predict(x_scaled)
                print(f"Logistic regression using [{features_list[i+1]}, {features_list[j+1]}, {features_list[k+1]}] features:\n%s\n" % (
                    metrics.classification_report(expected, predicted)))
                print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
                # estimate the generalization performance using cross validation
                predicted2 = cross_val_predict(logistic_classifier, x_scaled, y, cv=10)
                print(f"Logistic regression using {features_list[i+1]}, {features_list[j+1]} {features_list[k+1]} features cross validation:\n%s\n" % (
                    metrics.classification_report(expected, predicted2)))
                print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted2))
              
                
              
# data load
digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

''' SECTION 19 '''
misclassification()


'''  SECTION 20-C '''
# each variable is list that the feature function had return
sum_matrix = run_feature(digits.images, get_mat_sum)
variance_rows = run_feature(digits.images, get_rows_sum_var)
variance_columns = run_feature(digits.images, get_cols_sum_var)
main_area_sum = run_feature(digits.images, get_main_area_sum)
vertical_symmetric = run_feature(digits.images, get_measure_vertically_symmetric)
horizontal_symmetric = run_feature(digits.images, get_measure_horizontally_symmetric)


'''  SECTION 20-D '''
# select the data with value 0 | 1.
indices_0_1 = np.where(np.logical_and(digits.target >= 0, digits.target <= 1))
selected_data = digits.target[indices_0_1]


'''  SECTION 20-E '''
# define lists of data
features_list = [sum_matrix, "sum of matrix",
                 variance_rows, "variance of rows sum",
                 variance_columns, "variance of columns sum",
                 vertical_symmetric, "sum of main area",
                 horizontal_symmetric, "measure of vertical symmetric",
                 main_area_sum, "measure of horizontal symmetric"]

#define figure data
colors = ['yellow' if yi == 1 else 'purple' for yi in selected_data]

scatter_one_feature()
scatter_two_features()
scatter_three_features()


'''  SECTION 20-F '''    
logistic_regression_clf()


''' SECTION 20-G  '''
logistic_regression_clf_for_three()


