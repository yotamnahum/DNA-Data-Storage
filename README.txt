READ ME:

configuration.py: Is the files which sets and generates the configurations for the model, i.e the config.json file

data_utils: Contains data related methods: conversion and loading of datasets, and train_test split method

DNAmodel.py: This is the DL model. The file contatins the class DnaModel with initiation method, data process for the model, prediction methods and evaluation metrics

image_utils.py: Contains all relevant method for converting an image file\s to DNA strings, and reconstruct the original image from the DNA strings

imagenetkaggle.py: import imagenet from kaggle

Predict.py: This file holds prediction methods and evaluation of the results. Before prediction, there are several method for filtering and organizing the data according to CL,LL,SL etc...

text_utils.py: Contains all relevant method for converting an text file\s to DNA strings, and reconstruct the original text from the DNA strings

utils.py: Genral utils: evaluation methods, Error simulation method