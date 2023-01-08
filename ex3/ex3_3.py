"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)
.. _LFW: http://vis-www.cs.umass.edu/lfw/



                       Noam Mirjani 315216515
                      Yaakov Haimoff 318528510

"""
# %%
from time import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
from sklearn.metrics import precision_score


'''                   section 3A                     '''
def plot_pca_var():
    ''' plots the variance of the pca (multiple by 100) '''
    plt.plot(pca.explained_variance_ratio_ * 100)
    plt.xlabel('PC #')
    plt.ylabel('Variance [%]')
    plt.title("Variance vs. PC #")
    plt.show()
    
'''                   section 3B                     '''
def measureFit(clf):
    ''' measure the fit time and prints it.
        parameter clf return value void              '''
    start = time()
    clf = clf.fit(X_train_pca, y_train)
    end =   time()
    print("fitting time: ", end-start)
    
def precision():
    ''' measure the precision value and prints it'''
    precision = precision_score(y_test, y_pred, average='micro')
    print("Precision:", precision)


'''                   section 3E                    '''
def test_svm_clf():
    ''' measure the time of fitting  and the precision score of 
        classifier SVC linear and rbf '''
    start = time()
    clf_linear = RandomizedSearchCV(
        SVC(kernel="linear", class_weight="balanced"), param_grid, n_iter=10)
    end =   time()
    clf_linear = clf_linear.fit(X_train, y_train)
    predictions_linear  = clf_linear.predict(X_test)
    precision_score_linear = precision_score(y_test, predictions_linear, average='macro')
    print("LINEAR TIME: ", end-start, "precision", precision_score_linear)


    start = time()
    clf_rbf = RandomizedSearchCV(
        SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10)
    end =   time()
    clf_rbf = clf_rbf.fit(X_train, y_train)
    predictions_rbf  = clf_rbf.predict(X_test)
    precision_score_rbf = precision_score(y_test, predictions_rbf, average='macro')
    print("RBF TIME: ", end-start, "precision", precision_score_rbf)


# %%
# Qualitative evaluation of the predictions using matplotlib

'''                   section 3C                      '''
def plot_gallery(images, titles, h, w, n_row=7, n_col=7):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.45 * n_col, 1.5 * n_row))
    plt.subplots_adjust(bottom=0.03, left=.01, right=.99, top=.93,hspace=.36)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# %%
# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(" ", 1)[-1]
    true_name = target_names[y_test[i]].rsplit(" ", 1)[-1]
    return "predicted: %s\ntrue:      %s" % (pred_name, true_name)


# %%
# Download the data, if not already on disk and load it as numpy arrays
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)


# %%
# Split into a training set and a test and keep 25% of the data for testing.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


param_grid = {
    "C": loguniform(1e3, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}

'''                   section 3E                        '''
test_svm_clf()
'''                  end section 3E                        '''

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# %%
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

n_components = 150

print(
    "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
)
t0 = time()
pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
print("done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))

'''                 section 3A                       '''
plot_pca_var()
'''              end section 3A                       '''

print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))


# %%
# Train a SVM classification model

print("Fitting the classifier to the training set")
t0 = time()
clf = RandomizedSearchCV(
    SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
)

'''                section 3B1                           '''
measureFit(clf)
'''               end section 3B1                        '''

print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)


# %%
# Quantitative evaluation of the model quality on the test set

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
ConfusionMatrixDisplay.from_estimator(
    clf, X_test_pca, y_test, display_labels=target_names, xticks_rotation="vertical"
)

'''                section 3B2                         '''
precision()
'''               end section 3B2                        '''

plt.tight_layout()
plt.show()


prediction_titles = [
    title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])
]

plot_gallery(X_test, prediction_titles, h, w)

# %%
# plot the gallery of the most significative eigenfaces
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()


