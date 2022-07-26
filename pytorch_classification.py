from tkinter import Image
from unicodedata import category
from torch.utils.data import Dataset, DataLoader
from typing import Union

import pandas as pd
import torch

class ImageDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
    

    def __getitem__(self, index) -> Union[pd.DataFrame, pd.Series]:
        example = self.data.iloc[index]
        features = torch.tensor(example['image_array'])
        label = example['category']
        
        return features, label


    def __len__(self) -> int:
        return(len(self.data))


    def encode_categories(self) -> None:
        self.category_dict = dict(enumerate(self.data.category.cat.codes))
        self.data['category'] = self.data.category.cat.codes

        return None


    def decode_categories(self) -> None:
        self.data['category'] = self.data.category.map(self.category_dict)

        return None



# load pickle file into new dataframe
full_image_dataframe = pd.read_pickle('images_data.pkl')
image_dataframe = full_image_dataframe[['image_array', 'category']]

dataset = ImageDataset(image_dataframe)
dataset.encode_categories() # encodes category column
dataloader = DataLoader(dataset, batch_size=64)

print(next(iter(dataloader)))

#dataset.decode_categories() # decode category column