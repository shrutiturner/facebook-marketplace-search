import os
import pickle
import json

import uvicorn
import boto3
import botocore
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from PIL import Image

app = FastAPI()


bucket_path = os.environ.get("BUCKET_NAME")

s3 = boto3.resource('s3')
bucket = s3.Bucket(bucket_path)


@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/text')
def predict_text(body):
  try:
    bucket.download_file("text_model.pkl", "/home/ubuntu/text_model.pkl")
  except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == '404':
      print("Error downloading model, model doesn't exist.")
  else:
    raise
    
  with open('/home/ubuntu/text_model.pkl', 'rb') as text_model:
    model = pickle.load(text_model)
  """Below is an example"""
  # data = np.array(body)
  
  # probabilities = model.predict_prob(data)
  # predicitons = probabilities.argmax(axis=1)
  
  ##############################################################
  # TODO                                                       #
  # change the request body into an array/tensor and use your  #
  # models methods to run predictions                          #
  ##############################################################
    
  # return {
  #   "Predictions": "", #return predictions here
  #   "Probabilities": "" #return probabilities here
  #     }
  
  return {
    "Predictions": np.linspace(0, 2, 100), #return predictions here
    "Probabilities": "" #return probabilities here
      }
  
  
@app.post('/predict/image')
def predict_image(image):
  try:
    bucket.download_file("image_model.pkl", "/home/ubuntu/image_model.pkl")
  except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == '404':
      print("Error downloading model, model doesn't exist.")
  else:
    raise
    
  with open('/home/ubuntu/image_model.pkl', 'rb') as image_model:
    model = pickle.load(image_model)
    
  ##############################################################
  # TODO                                                       #
  # change the request body into an array/tensor and use your  #
  # models methods to run predictions                          #
  ##############################################################
    
  return {
    "Predictions": "", #return predictions here
    "Probabilities": "" #return probabilities here
      }
  
    
@app.post('/predict/combined')
def predict_combined(body):
  try:
    bucket.download_file("combined_model.pkl", "/home/ubuntu/combined_model.pkl")
  except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == '404':
      print("Error downloading model, model doesn't exist.")
  else:
    raise
    
  with open('/home/ubuntu/combined_model.pkl', 'rb') as combined_model:
    model = pickle.load(combined_model)
    
  ##############################################################
  # TODO                                                       #
  # change the request body into an array/tensor and use your  #
  # models methods to run predictions                          #
  ##############################################################
    
  return {
    "Predictions": "", #return predictions here
    "Probabilities": "" #return probabilities here
      }
    
    
if __name__ == '__main__':
  uvicorn.run("api:app", host="0.0.0.0", port=8080)