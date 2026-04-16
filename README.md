Problem Statement
You are given a small passenger dataset. Your task is to apply categorical encoding and feature scaling to prepare it for a machine learning model.

import pandas as pd

data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'City': ['Mumbai', 'Delhi', 'Chennai', 'Mumbai', 'Delhi'],
    'Size': ['Small', 'Large', 'Medium', 'Large', 'Small'],
    'Age': [25, 45, 32, 60, 28],
    'Fare': [15, 300, 85, 450, 20]
}

df = pd.DataFrame(data)


Task 1 — Categorical Encoding
Apply the correct encoding method to each categorical column:

Gender → Label Encoding
City → One-Hot Encoding (drop one column to avoid the dummy variable trap)
Size → Label Encoding with the correct order: Small=0, Medium=1, Large=2


Task 2 — Feature Scaling
The Fare column contains a value of 450 which is a potential outlier. Apply the appropriate scaler to both Age and Fare, and briefly justify your choice in a comment.

