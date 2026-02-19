"""
In this file, we implement the training of a logistic regression model and linear SVM.
We train the logistic model similiar to practical 1 and we use new sklearn functionality for the linear SVM.
The source of the sklearn documentation we looked up is below:
    LinearSVC implementation: https://shorturl.at/JtR7Z
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def train_log_reg(X_train, y_train, seed=7): #We train a logistic regression model using TD-IDF features.
    #We set a seed for reproducibility and max iterations of 1000, which was a typical number we set in previous ML projects.
    model = LogisticRegression(random_state=seed, max_iter=1000) 
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train, seed=7): #Here we train a linear SVM which is used as a secondary baseline.
    #Here, we instantiate the model. 
    #The dual auto parameter allows python to automatically choose which way to optimize according to the data shape.
    model = LinearSVC(random_state=seed, dual='auto') 
    model.fit(X_train, y_train)
    return model