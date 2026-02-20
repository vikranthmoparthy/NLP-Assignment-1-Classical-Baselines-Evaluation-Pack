"""
This file implements the main pipeline, combining all functionality from the python files under the source_code folder.
There isn't much new functionality in this file, but we looked a little deeper into sklearn's TfidfVectorizer.
    sklearn's TfidfVectorizer: https://short-url.org/1p83s
"""

from source_code.data_loader import load_and_split_data
from source_code.preprocessing import preprocess_dataframe
from source_code.models import train_log_reg, train_svm
from source_code.evaluation import evaluate_model, plot_confusion_matrix, get_misclassified_examples
from sklearn.feature_extraction.text import TfidfVectorizer

RAND_SEED = 7
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

def evaluate_logregmodel(x_train_vec, y_train, x_dev_vec, y_dev, x_test_vec, y_test):
    #We now train the logistic regression model using the vectorized training data.
    lr_model = train_log_reg(x_train_vec, y_train, seed=RAND_SEED) 
    
    #report metrics for both dev and test sets
    print("Logistic Regression Develeopment results:")
    evaluate_model(lr_model, x_dev_vec, y_dev)
    
    print("Logistic Regression Test results:")
    evaluate_model(lr_model, x_test_vec, y_test)


def evaluate_svm_plus_errors(x_train_vec, y_train, x_dev_vec, y_dev, x_test_vec, y_test, x_test_text):
    #Now, we initialize the linear SVM model
    svm_model = train_svm(x_train_vec, y_train, seed=RAND_SEED)
    
    #report metrics for SVM
    print("SVM Development results:")
    evaluate_model(svm_model, x_dev_vec, y_dev)
    
    print("SVM Test Results:")
    y_pred_svm = evaluate_model(svm_model, x_test_vec, y_test)     #store predictions for error analysis

    errors_df = get_misclassified_examples(x_test_text, y_test, y_pred_svm, label_names=LABEL_NAMES) #Identify incorrect SVM examples

    errors_df.to_csv("svm_errors.csv", index=False) #stores errors (see: svm_errors.csv)

    plot_confusion_matrix(y_test, y_pred_svm, label_names=LABEL_NAMES) #Finally, we display the confusion matrix


def main():
    df_train, df_dev, df_test = load_and_split_data(seed=RAND_SEED) #Fetch the data and create the splits

    #Clean and preprocess the text
    x_train_text, y_train = preprocess_dataframe(df_train)
    x_dev_text, y_dev = preprocess_dataframe(df_dev)
    x_test_text, y_test = preprocess_dataframe(df_test)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) #Create tf-idf Vectorizer
    
    #We fit only on the training set, then transform the Dev and Test sets
    x_train_vec = vectorizer.fit_transform(x_train_text)
    x_dev_vec = vectorizer.transform(x_dev_text)
    x_test_vec = vectorizer.transform(x_test_text)
    
    evaluate_logregmodel(x_train_vec, y_train, x_dev_vec, y_dev, x_test_vec, y_test)
    evaluate_svm_plus_errors(x_train_vec, y_train, x_dev_vec, y_dev, x_test_vec, y_test, x_test_text)

if __name__ == "__main__":
    main()