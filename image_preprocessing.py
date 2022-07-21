from clean_tabular_data import get_and_normalise_data

import os
import pandas as pd

from numpy import asarray, product
from PIL import Image

file_path = "Products.csv"
lineterminator = "\n"

product_categories = get_and_normalise_data(file_path, lineterminator)[['id','category']]
product_categories.rename(columns={'id': 'product_id'}, inplace=True)

# encode catgeory column
product_categories['category'] = product_categories.category.cat.codes

# transform images into numpy array
path = "../cleaned_images/"
dirs = os.listdir(path)

image_array_df = pd.DataFrame(columns = ['image_id', 'image_array'])

for n, item in enumerate(dirs[:], 1): # change index here to limit number of images processed
    try:
        image = Image.open(path + item)
        data = asarray(image)

        image_array_df.loc[n-1] = [item[:-4], data]
    
    except Exception as e:
        print(f'{item} can not be converted to nparray.')
        print(e)

# merge numpy array and category column(s) into one dataframe
image_data = None


# save dataframe as pickle file
image_data.to_pickle('image_data.pkl')