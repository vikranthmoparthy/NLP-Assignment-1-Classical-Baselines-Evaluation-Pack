"""
In this file, we implement preprocessing. We normalize text by lowercasing it and removing non-alphanumeric characters.
We combine the title and description columns in a dataframe, and return the processed text with its labels for training.
To write this code, we needed to look up some pandas and regex documentation.
    Pandas column concatenation https://shorturl.at/yrMS2
    re.sub(): https://shorturl.at/jnUyq
    Pandas apply() function: https://shorturl.at/FR0QQ
"""

import re
import pandas as pd

def clean_text(text): #Function to converting string to lowercase
    if not isinstance(text, str):
        return ""
    text = text.lower()     

    #Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip() #Replace multiple spaces with a single space, and also strip edges.

    return text

def preprocess_dataframe(df): #This function prepares the dataframe by combining text fields and cleaning
    #We concatenate columns of title and description with a single space.
    combined_text = df['title'].astype(str) + " " + df['description'].astype(str)
    x = combined_text.apply(clean_text) #We use pandas apply method to clean every row
    y = df['label']

    return x, y