# Standard scientific Python imports
import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#               FEATURE 1  - SUM MATRIX
# calculate the sum of each image's pixel values, return array that each cell
#  has the sum of the matrix
# return value sums (list)
def sum_of_mat(digits_images):
    sums = []
    for image1 in digits_images:
        sums.append(image1.sum())
    return sums


#               FEATURE 2  - VARIANCE OF ROWS SUM
# function iterates over the rows of each image in digits_images, calculate
# the variance to each row's, and appends the result to the list called sums.
# parameters -> digits_images(matrix to iterate over)
# return value -> sums (list)
def rows_calc_variance(digits_images):
    sums = []
    for image1 in digits_images:
        sum_of_row = 0
        for row in image1:
            sum_of_row += row
        s = np.var(sum_of_row)
        sums.append(s)
    return sums


#               FEATURE 3  - VARIANCE OF COLS SUM
# function iterates over the rows of each image in digits_images, calculate
# the variance to each row's, and appends the result to the list called sums.
# To willing iterate over the columns, transport the matrix, and run over the rows
# parameters ->  digits_images(matrix to iterate over)
# return value -> sums (list)
def cols_calc_variance(digits_images):
    sums = []
    for image1 in digits_images:
        image1 = np.transpose(image1)
        sum_of_col = 0
        for col in image1:
            sum_of_col += col
        s = np.var(sum_of_col)
        sums.append(s)
    return sums


#              FEATURE 4 - SUM OF MAIN AREA IN MATRIX
# The images are in six[8,8], the main aria found in rows 3,4 cols 3,4
# parameters -> digits_images(matrix to iterate over)
# return value -> sums (list)
def sum_of_main_area(digits_images):
    sums = []
    for img in digits_images:
        temp_sum = 0
        for i in range(3, 5):
            for j in range(3, 5):
                temp_sum += img[i][j]
        sums.append(temp_sum)
    return sums


#              FEATURE 5,6 - PERCENTAGE OF SYMMETRIC MATRIX
# parameters -> digits_images(matrix to iterate over) and func to run
# return value -> retArr (list)
def symmetric(digits_images, is_symmetric_func):
    ret_arr = []
    for img in digits_images:
        ret_arr.append(is_symmetric_func(img))

    return ret_arr


# Checking for Vertical Symmetry. We compare
# first column with last column, second column
# with second last column and so on.
def is_vertically_symmetric(matrix):
    counter_symmetric = 0
    j = 0
    k = len(matrix) - 1
    while j < len(matrix) // 2:
        for i in range(len(matrix)):
            if matrix[i][j] != matrix[i][k]:
                break
        else:
            counter_symmetric += 2
        j += 1
        k -= 1
    return counter_symmetric / len(matrix)


# Checking for Horizontal Symmetry. We compare
# first row with last row, second row with second
# last row and so on.
def is_horizontally_symmetric(matrix):
    counter_symmetric = 0
    i = 0
    k = len(matrix) - 1
    while i < len(matrix) // 2:
        for j in range(len(matrix)):
            if matrix[i][j] != matrix[k][j]:
                break
        else:
            counter_symmetric += 2
        i += 1
        k -= 1
    return counter_symmetric / len(matrix)


# data load
digits = datasets.load_digits()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# each variable is list that the feature function had return
sum_matrix = sum_of_mat(digits.images)
variance_rows = rows_calc_variance(digits.images)
variance_columns = cols_calc_variance(digits.images)
main_area_sum = sum_of_main_area(digits.images)
vertical_symmetric = symmetric(digits.images, is_vertically_symmetric)
horizontal_symmetric = symmetric(digits.images, is_horizontally_symmetric)

# select the data with value 0 | 1.
indices_0_1 = np.where(np.logical_and(digits.target >= 0, digits.target <= 1))
selected_data = digits.target[indices_0_1]

# add the sums as a new feature to the data array
data = np.column_stack((data, sum_matrix))


# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
    data[indices_0_1], digits.target[indices_0_1], test_size=0.5, random_state=42, shuffle=False)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

predict = clf.predict(X_test)
figure = plt.figure(figsize=(10, 10))
ax = plt.subplot()
ax.scatter(y_test, predict)
# ax.set_xlim(sum_matrix)
# ax.set_ylim(sum_matrix)

plt.tight_layout()
plt.show()