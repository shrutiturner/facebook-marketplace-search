from datetime import datetime
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

import clean_tabular_data

class ImageDataset(Dataset):
    """Creates tensor image dataset for use in model.

    Args:
        Dataset (Dataset): Pytorch class
    """
    def __init__(self, data_file='api/image_cats.json', root_dir='/home/ubuntu/', image_folder='cleaned_images', transform = None) -> None:
        super().__init__()

        #clean and merge tabular data
        self.prod_data = clean_tabular_data.get_and_normalise_data("api/Products.csv", '\n')
        self.image_ids = pd.read_csv ("api/Images.csv")
        self.merged_data = self.image_ids.merge(self.prod_data[['category','product_id']], on='product_id') #not all data is kept in the data!
        
        #sets labels and image 
        self.labels = self.merged_data['category'].to_list()
        self.image_files = self.merged_data['id'].to_list()

        #encode labels
        self.encoded_labels = {}
        self.encode_labels(self.merged_data)


    def __getitem__(self, index) -> Union[pd.DataFrame, pd.Series]:
        label = self.labels[index]
        encoded_label = torch.tensor(self.encoded_labels[label])
        image = self.image_files[index]
        try:
            PIL_image = Image.open(f"cleaned_images/{image}.jpg")   
            transform = transforms.PILToTensor() 
            feature = transform(PIL_image)
            feature = torch.flatten(feature).to(torch.float32)
            # print (feature.shape)
            # print ("LABEL IS")
            # print (encoded_label)
            return feature, encoded_label
        except Exception as e:
            print(e)
        

    def __len__(self) -> int:
        return len(self.merged_data)


    def encode_labels(self, merged_data) -> None: #435 labels total
        """Encodes labels to integers.

        Args:
            merged_data (pd.DataFrame): DataFrame of input data about images.

        Returns:
            None: Returns None type.
        """
        full_catagories = merged_data['category'].unique()
        
        for cat in enumerate (full_catagories):
            self.encoded_labels[cat[1]] = cat [0]
        
        return None


    def get_category(self, label_index) -> dict:
        
        return self.encoded_labels[label_index]


class CNN(torch.nn.Module):
    """Creates a convolutional neural network using transfer learning.

    Args:
        torch (torch.nn.Module): Pytorch neural network module.
    """
    def __init__(self) -> None:
        super().__init__()
        
        self.features = models.resnet50(pretrained=True)
        for i, param in enumerate(self.features.parameters()):
            if i < 47:
                param.requires_grad=False
            else:
                param.requires_grad=True
        self.features.fc = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024), # first arg is the size of the flattened output from resnet50
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 128),
            torch.nn.ReLU(),
            torch.nn.Linear((128), 13)
            )


    def forward(self, inp) -> transforms.ToTensor:
        """Input manipulation in the forwards direction of neural network.

        Args:
            inp (torchvision.tensor): input tensor of data.

        Returns:
            tensor: output tensor for neural network input.
        """
        inp = inp.reshape(inp.shape[0], 3, 64, 64)
        x = self.features(inp)
        return x


def train(model, model_time, epochs=10) -> None:
    """Trains model inputted in epochs.

    Args:
        model (torch.nn.models): Input neural network model to be trained.
        epochs (int, optional): number of epochs to train model. Defaults to 10.

    Returns:
        None: None type is returned.
    """
    
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter()
    batch_idx = 0

    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader, 0):
            features, labels = batch

            optimiser.zero_grad()

            prediction = model(features)
            loss = criterion(prediction, labels.long())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad() #this reverts the grad value to zero so .backwards will overwright (otherwise it would just add to the grad val)

            #print(f'Loss: {loss.item()}')

            running_loss += loss.item()
            if i % 20 == 19:
                print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss / 2000:3f}')
                running_loss = 0.0

            
            writer.add_scalar('Loss', loss.item(), batch_idx)
            batch_idx += 1
        
        torch.save(model.state_dict(), f'api/model_evaluation/{model_time}/weights/epoch_{epoch}.pt')
    
    return None


if __name__ == '__main__':
    dataset = ImageDataset()
    
    dataloader = DataLoader(dataset, shuffle=True, batch_size=8)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CNN()

    model_time = datetime.now()

    train(model, model_time)

    # save model
    state_dict = torch.save(model.state_dict(), f'api/final_models/{model_time}/image_model.pt')

    # state_dict = torch.load('model.pt')
    # new_model = CNN()
    # new_model.load_state_dict(state_dict)
    # train(new_model)


