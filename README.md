# facebook-marketplace-search
Capstone project for Ai Core - creating a replica of the Facebook Marketplace Search Ranking

AiCore project to demonstrate Machine Learning and Machine Learning Engineering skills, based on the algorithms used for the Facebook Marketplace Search Algorithm.

This README documents the actions taken and decisions made during each step of the project.

Technologies used: Python (boto3, fastapi, math, numpy, pandas, Pillow, pydantic, python-multipart, pytz, transformers, uvicorn)

## Milestone 1 - Overview of the System
Watched a video to understand the goals of the project.

## Milestone 2 - Explore the Dataset
Connected to the provided sandbox environment. The environment was set up in the EC2 instance using the content in the S3 buckets provided. Tabular data was provided about the products in a csv file. A pipleine was created, using pandas, numpy and math, to remove rows with missing data and clean the tabular data - a mix of numberical, categorical, text and dates. A second pipeline was created to clean the image data provided. Copies of the image were created with consistent dimensions and mode (RGB was chosen).

## Milestone 3 - Create Simple Machine Learning Models
Two simples models were created, a regression model to predict the price of products based on tabular data and a classification model to categorise product images associated with products. The models required manipulation of the cleaned data sets to fit with the models, including encoding of categorical variables. The models perform poorly, but they are an example of pipelines/implemented methods to creat them. If anything, using these to predict target variables shows the neccessity of deep learning compared to machine learning. The models could, of course be improved: for the regression model, more of the columns could be manipulated to be used as features and for the classification model the images could have been normalised to be larger. With such small images (64x64 pixels) much of the image detail is lost. Other models could also have been used, including ensemble methods, as well as hyperparameter tune and cross-validation.
