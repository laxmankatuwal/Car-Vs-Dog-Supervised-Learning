#importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Create synthetic dataset
np.random.seed(0)
head_size = np.random.normal(30, 5, 100) 
body_length = np.random.normal(50, 10, 100) 
labels = np.random.choice([0, 1], 100)  

# Combine into a DataFrame
data = pd.DataFrame({
    'head_size': head_size,
    'body_length': body_length,
    'label': labels
})

# Train logistic regression model
X = data[['head_size', 'body_length']]
y = data['label']
model = LogisticRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'cat_dog_classification_model.pkl')

