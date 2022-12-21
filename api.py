import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from torchvision import models
from pydantic import BaseModel

from image_processor import image_processor
from pytorch_classification import CNN, train


class CombinedModel(nn.Module):
    def __init__(self,
                 decoder: list = None):
        super(CombinedModel, self).__init__()

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
        
        self.decoder = decoder

    def forward(self, image_features, text_features):
        pass

    def predict(self, image_features, text_features):
        with torch.no_grad():
            combined_features = self.forward(image_features, text_features)
            return combined_features
    
    def predict_proba(self, image_features, text_features):
        with torch.no_grad():
            pass

    def predict_classes(self, image_features, text_features):
        with torch.no_grad():
            pass



# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str

try:
    combined_model = CNN()
    state_dict = torch.load('model.pt')
    combined_model.load_state_dict(state_dict)
    train(combined_model)

    with open('combined_decoder.pkl', 'rb') as f:
            image_decoder = pickle.load(f)
except:
    raise OSError("No Combined model found. Check that you have the encoder and the model in the correct location")

try:
    with open('image_decoder.pkl', 'rb') as f:
        image_decoder = pickle.load(f)
except:
    raise OSError("No Image processor found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}


@app.post('/predict/combined')
def predict_combined(image: UploadFile = File(...), text: str = Form(...)):
    print(text)

    img = Image.open(image.file)
    processed_img = image_processor(img)
    prediction = combined_model(processed_img)

    return JSONResponse(content={
    "Category": "", # Return the category here
    "Probabilities": "" # Return a list or dict of probabilities here
        })
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)