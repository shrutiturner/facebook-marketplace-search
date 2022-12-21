from regex import R
from clean_tabular_data import get_and_normalise_data

import os
import numpy as np
import pandas as pd

from numpy import asarray, product
from PIL import Image

file_path = "../api/Products.csv" 
lineterminator = "\n"

product_categories = get_and_normalise_data(file_path, lineterminator)
product_categories.rename(columns={'id': 'product_id'}, inplace=True)
#product_categories.rename(columns={'Unnamed: 0': 'entry_index'}, inplace=True)

product_id_cats = product_categories[['product_id', 'category']]


# encode catgeory column
#product_categories['category'] = product_categories.category.cat.codes

# import info from images.csv

images_data = pd.read_csv("Images.csv")
images_data = images_data[['id', 'product_id']]
images_data.rename(columns={'id': 'image_id'}, inplace=True)

# merge numpy array and category column(s) into one dataframe
combined_df = pd.merge(images_data, product_categories, on='product_id')

print(combined_df[['image_id', 'category']].head())
#print(combined_df['image_array'].shape == (0,))

# save dataframe as pickle file
combined_df[['image_id', 'category']].to_json('image_cats.json')