from tkinter import Image
from unicodedata import category
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from typing import Union

import pandas as pd
import torch
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.data.category = self.encode_categories(data.category)
        print(self.data.category.value_counts())
    

    def __getitem__(self, index) -> Union[pd.DataFrame, pd.Series]:
        example = self.data.iloc[index]
        features = torch.tensor(example['image_array'])
        label = example['category']
        
        return features, label


    def __len__(self) -> int:
        return(len(self.data))


    def encode_categories(self, category_col: pd.Series) -> None:
        self.category_dict = dict(enumerate(category_col.cat.codes))
        category_col = category_col.cat.codes

        return category_col


    def decode_categories(self) -> None:
        self.data['category'] = self.data.category.map(self.category_dict)

        return None


def train(model, epochs=10):
    
    optimiser = torch.optim.SGD(model.parameters(), lr=0.1)

    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in dataloader:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction.long, labels.long())
            loss.backward()
            print(f'Loss: {loss.item()}')
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scaler('Loss', loss.item(), batch_idx)
            batch_idx += 1


class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(64, 7, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 16, 7),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(234, 13),
            torch.nn.ReLU(),
            torch.nn.Softmax()
        )
    
    def forward(self, X):
        return self.layers(X)


if __name__ == '__main__':
    # load pickle file into new dataframe
    image_dataframe = pd.read_pickle('images_data.pkl')
    #image_dataframe = full_image_dataframe[['image_array', 'category']]

    print(image_dataframe.dtypes)

    dataset = ImageDataset(image_dataframe)
    #dataset.encode_categories() # encodes category column
    dataloader = DataLoader(dataset, shuffle=True, batch_size=64)

    model = CNN()

    train(model)

    #dataset.decode_categories() # decode category column