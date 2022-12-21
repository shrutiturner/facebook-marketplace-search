import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load pickle file into new dataframe
image_dataframe = pd.read_pickle('images_data.pkl')

image_dataframe['image_array'] = image_dataframe['image_array'].apply(lambda x: x.flatten())


# features
X = list(image_dataframe['image_array'])

# target
y = list(image_dataframe['category'])

# pipeline?

# parameters?

# split data in to train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# split test data into val/test
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

# instantiate and fit Logistic Regression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# predict
predictions = logistic_regression.predict(X_test)

# calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy Score: {accuracy}')
