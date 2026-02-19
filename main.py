"""
This file implements the main pipeline, combining all functionality from the python files under the source_code folder.
There isn't much new functionality in this file, but we looked a little deeper into sklearn's TfidfVectorizer.
    sklearn's TfidfVectorizer: https://short-url.org/1p83s
"""

from source_code.data_loader import load_and_split_data
from source_code.preprocessing import preprocess_df
from source_code.models import train_log_reg, train_svm
from source_code.evaluation import evaluate_model, plot_confusion_matrix, get_misclassified_examples
from sklearn.feature_extraction.text import TfidfVectorizer

#Random seed is set for reproducibility
RAND_SEED = 7
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

def main():
    df_train, df_dev, df_test = load_and_split_data(seed=RAND_SEED) #Fetch the data and create the splits

    #Clean and preprocess the text
    x_train_text, y_train = preprocess_df(df_train)
    x_dev_text, y_dev = preprocess_df(df_dev)
    x_test_text, y_test = preprocess_df(df_test)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000) #Initialize the TF-IDF Vectorizer
    
    #We fit ONLY on the training set to prevent data leakage.
    #We then transform the Dev and Test sets
    x_train_vec = vectorizer.fit_transform(x_train_text)
    x_dev_vec = vectorizer.transform(x_dev_text)
    x_test_vec = vectorizer.transform(x_test_text)
    
    #We now train the logistic regression model using the vectorized training data.
    lr_model = train_log_reg(x_train_vec, y_train, seed=RAND_SEED) 
    
    #report metrics for both dev and test sets
    print("Logistic Regression Dev Results:")
    evaluate_model(lr_model, x_dev_vec, y_dev)
    
    print("Logistic Regression Test Results:")
    evaluate_model(lr_model, x_test_vec, y_test)

    #Now, we initialize the linear SVM model
    svm_model = train_svm(x_train_vec, y_train, seed=RAND_SEED)
    
    #report metrics for SVM
    print("SVM Dev Results:")
    evaluate_model(svm_model, x_dev_vec, y_dev)
    
    print("SVM Test Results:")
    #we store predictions to use in the error analysis below
    y_pred_svm = evaluate_model(svm_model, x_test_vec, y_test)

    #Here, we identify examples where the SVM predicted incorrectly
    errors_df = get_misclassified_examples(x_test_text, y_test, y_pred_svm, label_names=LABEL_NAMES)

    #We store these errors in csv
    errors_df.to_csv("svm_errors.csv", index=False)

    plot_confusion_matrix(y_test, y_pred_svm, label_names=LABEL_NAMES) #Finally, we display the confusion matrix

if __name__ == "__main__":
    main()