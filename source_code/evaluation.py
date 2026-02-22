"""
This file implements model evaluation and error analysis for the classification. We define functions to compute accuracy, macro-F1, create
a confusion matrix and also extract incorrectly classified data into a separate csv for later inspection. 
Again, we had to consult some documentation for writing this code, specifically for calculating the evaluation metrics
    Macro f1-score: https://shorturl.at/grdu1
    ConfusionMatrixDisplay: https://shorturl.at/S6GFn
    Accuracy score: https://short-url.org/1p82c
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(model, x_features, y_true): #Function to compute accuracy and macro-f1
    y_predictions = model.predict(x_features)
    
    accuracy = accuracy_score(y_true, y_predictions)
    f1 = f1_score(y_true, y_predictions, average='macro')
    
    print(f"Accuracy: {accuracy}")
    print(f"Macro f1: {f1}")
    
    return y_predictions

#Function that generates and displays a confusion matrix to interpret misclassification patterns.
def plot_confusion_matrix(y_true, y_predictions, label_names=None):
    conf_matrix = confusion_matrix(y_true, y_predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_names)
    #We use matplotlib for better visualization
    figure, axes = plt.subplots(figsize=(10, 10))
    display.plot(cmap='Blues', ax=axes, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

#Function that extracts / organizes samples where the model predicted the wrong category.
def get_misclassified_examples(x_text, y_true, y_predictions, label_names=None): 
    #Here, we convert numeric class IDs back to names e.g."Sports".
    if label_names is not None:
        try:
            y_true = [label_names[i] for i in y_true]
            y_predictions = [label_names[i] for i in y_predictions]
        except (TypeError, IndexError):
            pass
    
    #We create a dataframe view text alongside its labels.
    df_predictions = pd.DataFrame({'text': x_text, 'true_label': y_true,'prediction_label': y_predictions})
    
    #We identify errors by comparing predicted label to ground truth label.
    errors = df_predictions[df_predictions['true_label'] != df_predictions['prediction_label']]
    
    print(f"No. of Errors: {len(errors)}")
    
    return errors