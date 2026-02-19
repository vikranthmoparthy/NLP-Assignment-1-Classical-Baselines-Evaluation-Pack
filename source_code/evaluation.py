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
    y_pred = model.predict(x_features)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    
    return y_pred

#Function that generates and displays a confusion matrix to interpret misclassification patterns.
def plot_confusion_matrix(y_true, y_pred, label_names=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    #We use matplotlib for better visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

#Function that extracts and organizes samples where the model predicted the wrong category
def get_misclassified_examples(x_text, y_true, y_pred, label_names=None): 
    #Here, we convert numeric class IDs back to readable names e.g."Sports".
    if label_names is not None:
        try:
            y_true = [label_names[i] for i in y_true]
            y_pred = [label_names[i] for i in y_pred]
        except (TypeError, IndexError):
            pass
    
    #We create a dataframe to filter and view text alongside its labels.
    df_predictions = pd.DataFrame({'text': x_text, 'true_label': y_true,'pred_label': y_pred})
    
    #We identify errors by comparing predicted label to ground truth label.
    errors = df_predictions[df_predictions['true_label'] != df_predictions['pred_label']]
    
    print(f"Total Errors: {len(errors)}")
    
    return errors