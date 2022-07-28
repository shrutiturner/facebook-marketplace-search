from PIL import Image
from unicodedata import category
from numpy import asarray
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models
from typing import Union

import json
import os
import pandas as pd
import torch
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, data_file='api/image_cats.json', root_dir='/home/ubuntu/', image_folder='cleaned_images', transform = None) -> None:
        super().__init__()

        self.root_dir = root_dir
        self.image_data = pd.read_json(f'{root_dir}{data_file}')
        self.image_data['category'] = self.encode_categories(self.image_data['category'])
        self.image_folder = image_folder
        self.transform = transform


    def __getitem__(self, index) -> Union[pd.DataFrame, pd.Series]:
        try:
            image_path = os.path.join(f'{self.root_dir}{self.image_folder}/{self.image_data.iloc[index,0]}.jpg')
            features = imread(f'{image_path}')
            features = torch.tensor(features).float()
            features = features.reshape(3, 64, 64)

            labels = torch.tensor(self.image_data.iloc[index,1])

            if self.transform:
                features = self.transform(features)
            
            return features, labels
            

        except Exception as e:
            print(e) # errors expected as initially any rows with missing data were dropped from the dataframe
            pass
        

    def __len__(self) -> int:
        return(len(self.image_data))


    def encode_categories(self, category_col: pd.Series) -> None:
        category_col = category_col.astype('category')

        decoder_dict = dict(enumerate(category_col.cat.codes))

        with open('image_decoder.json', 'w') as file:
            json.dump(decoder_dict, file)
        
        category_col = category_col.cat.codes
        
        return category_col


    def decode_categories(self) -> None:
        with open('image_decoder.json', 'r') as file:
            decoder = json.load(file)

        self.data['category'] = self.data.category.map(decoder)
        return None


def train(model, epochs=10):
    
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        for batch in dataloader:
            features, labels = batch
            prediction = model(features)
            loss = F.cross_entropy(prediction, labels.long())
            loss.backward()
            print(f'Loss: {loss.item()}')
            optimiser.step()
            optimiser.zero_grad()
            writer.add_scalar('Loss', loss.item(), batch_idx)
            batch_idx += 1


class CNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.layers = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 45, 35),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(45, 12, 7),
        #     torch.nn.ReLU(),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(6912, 13),
        #     torch.nn.ReLU(),
        #     torch.nn.Softmax()
        # )

        model = models.resnet18(pretrained=True)
        num_features = model.fc.in_features

        model.fc = torch.nn.Linear(num_features, 13)

    
    def forward(self, X):
        return self.layers(X)


transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


if __name__ == '__main__':
    dataset = ImageDataset(transform=transform)
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=8)

    model = CNN()

    train(model)

    #dataset.decode_categories() # decode category column