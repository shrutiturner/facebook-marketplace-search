# facebook-marketplace-search
Capstone project for Ai Core - creating a replica of the Facebook Marketplace Search Ranking

AiCore project to demonstrate Machine Learning and Machine Learning Engineering skills, based on the algorithms used for the Facebook Marketplace Search Algorithm.

This README documents the actions taken and decisions made during each step of the project.

Technologies used: Python (boto3, fastapi, math, numpy, pandas, Pillow, pydantic, python-multipart, pytorch, pytz, transformers, uvicorn)

## Milestone 1 - Overview of the System
Watched a video to understand the goals of the project.

## Milestone 2 - Explore the Dataset
Connected to the provided sandbox environment. The environment was set up in the EC2 instance using the content in the S3 buckets provided. Tabular data was provided about the products in a csv file. A pipleine was created, using pandas, numpy and math, to remove rows with missing data and clean the tabular data - a mix of numberical, categorical, text and dates. A second pipeline was created to clean the image data provided. Copies of the image were created with consistent dimensions and mode (RGB was chosen).

## Milestone 3 - Create Simple Machine Learning Models
Two simples models were created, a regression model to predict the price of products based on tabular data and a classification model to categorise product images associated with products. The models required manipulation of the cleaned data sets to fit with the models, including encoding of categorical variables. The models perform poorly, but they are an example of pipelines/implemented methods to creat them. If anything, using these to predict target variables shows the neccessity of deep learning compared to machine learning. The models could, of course be improved: for the regression model, more of the columns could be manipulated to be used as features and for the classification model the images could have been normalised to be larger. With such small images (64x64 pixels) much of the image detail is lost. Other models could also have been used, including ensemble methods, as well as hyperparameter tune and cross-validation.

## Milestone 4 - Create the Vision Model
A milestone of iterative changes and improvements, showing how it is best to start with a starighforward model and then update and improve it once it works. The goal was to create and train a neural network to ingest images and classify the category the product belonged to. First the images retrieved were converted to a PyTorch dataset, i.e. in the form of tensors, to be ingested into a neural network. A simple convolutional neural network was created, along with a training loop which processed batches in the given number of epochs. To try and improve on the model, transfer learning was applied using a resnet50 model from the models library. Finally, to enable to the future used of a trained model, and monitoring of model performance, details and weights of the models were saved for each epoch and also the final model.

## Milestone 5 - Configure and Deploy the Model Serving API
***This does not function in the repo as the project instructions changed part way through completion, changing the requirements. The steps have been taken as a learning exercise and included only for these reason. All previous code runs, except the api.py script.***
A script was written to create an API using FastAPI to enable use of the model from any device. The goal was to create and API that any user can send a request to for the product to be categorised. The model is saved in a docker image inside a docker container, this is where the model will run. From the user's perspective, they send a request to the API and a response is returned with the category. No code runs locally.
