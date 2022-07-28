from regex import R
from api.clean_tabular_data import get_and_normalise_data

import os
import numpy as np
import pandas as pd

from numpy import asarray, product
from PIL import Image

file_path = "../api/Products.csv" 
lineterminator = "\n"

product_categories = get_and_normalise_data(file_path, lineterminator)[['id', 'price','category']]
product_categories.rename(columns={'id': 'product_id'}, inplace=True)

# encode catgeory column
product_categories['category'] = product_categories.category.cat.codes

# import info from images.csv

images_data = pd.read_csv("Images.csv")
images_data = images_data[['id', 'product_id']]
images_data.rename(columns={'id': 'image_id'}, inplace=True)

# transform images into numpy array
path = "../cleaned_images/"
dirs = os.listdir(path)

images_data['image_array'] = [np.empty((12288,), dtype=object)] * len(images_data)

for n, item in enumerate(dirs[:], 1): # change index here to limit number of images processed
    try:
        image = Image.open(path + item)
        data = asarray(image)

        images_data.loc[images_data['image_id'] == item[:-4], 'image_array'] = [data]


    except Exception as e:
        print(e) # errors expected as initially any rows with missing data were dropped from the dataframe
        pass

# merge numpy array and category column(s) into one dataframe
combined_df = pd.merge(images_data, product_categories, on='product_id')

#print(combined_df)
#print(combined_df['image_array'].shape == (0,))

# save dataframe as pickle file
combined_df.to_pickle('images_data.pkl')