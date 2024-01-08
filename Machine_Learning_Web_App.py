# import libraries
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# application heading
st.write("""
# Explore different Machine Learning Models and Datasets 
         to observe which one is best?
         """)

# store dataset names in a variable(dataset_name) and display them in the sidebar with dropdown menu
dataset_name = st.sidebar.selectbox(
    'Select Dataset', # name of the dropdown menu/sidebar
    ('Iris', 'Breast Cancer', 'Wine')
)

# store model names(classifier) in a variable(model_name) and display them in the sidebar with dropdown menu
model_name = st.sidebar.selectbox(
    'Select Classifier', # name of the dropdown menu/sidebar
    ('KNN', 'SVM', 'Random Forest')
)

# define a function to load the dataset
def get_dataset(dataset_name):
    if dataset_name == 'Iris':
        data = datasets.load_iris()
    elif dataset_name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

# Call the function
X, y = get_dataset(dataset_name)

# display the dataset shape
st.write("Shape of dataset:", X.shape)

# display the dataset number of classes
st.write("Number of classes:", len(np.unique(y)))


# define a function to add parameters of classifiers in the user inputs

def add_parameter_ui(classifier_name):
    params = dict()  # create an empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C  # its the degree of correct classification
    elif classifier_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K  # its the number of nearest neighbours
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth  # its the maximum depth of the tree in the random forest
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

# call the function
params = add_parameter_ui(model_name)

# define a function to build the model with the user inputs
def get_classifier(classifier_name, params):
    clf = None
    if classifier_name == 'SVM':
        clf = SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], max_depth=params['max_depth'], random_state=1234)
    return clf

# call the function
clf = get_classifier(model_name, params)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# train the model
clf.fit(X_train, y_train)

# predict the model
y_pred = clf.predict(X_test)

# display the accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write("Classifier = ", model_name)
st.write("Accuracy:", accuracy)

# plot the dataset

# we use PCA to reduce the dimensionality of the dataset to 2D, it reduce muliple features to two features
pca = PCA(2)
X_projected = pca.fit_transform(X)

# split data into 0 and 1 dimension slices
x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

# plot
fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# show the plot
st.pyplot(fig)