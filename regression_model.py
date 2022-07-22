import numpy as np

from clean_tabular_data import get_and_normalise_data

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer


file_path = "Products.csv"
lineterminator = "\n"

tabular_data = get_and_normalise_data(file_path, lineterminator)

# features
X = tabular_data[['product_name', 'product_description', 'location']]

# target
y = tabular_data['price']

# transform feature columns
transformer = ColumnTransformer(
    [('vect1', TfidfVectorizer(), 'product_name'),
     ('vect2', TfidfVectorizer(), 'product_description'),
     ('vect3', TfidfVectorizer(), 'location')],
    remainder='passthrough'
)

# create pipeline
pipeline = Pipeline(
    [
        ("colt", transformer),
        ("lr", LinearRegression())
    ]
)

# set parameters
parameters = {
    'colt__vect1__ngram_range': ((1, 1), (1, 2)),
    'colt__vect1__min_df': (0.005, 0.008, 0.01),
    'colt__vect2__ngram_range': ((1, 1), (1, 2), (1, 3)),
    'colt__vect2__min_df': (0.005, 0.008, 0.01),
}

# split data in to train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# split test data into val/test
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

# instantiate and fit grid_search with given parameters
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# predict with grid_search
predictions = grid_search.predict(X_test)

# calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f'RMSE: {rmse}')

