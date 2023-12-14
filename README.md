# Federated_Learning---MATCH_Software_Project
Building a federated learning system

Hello

Video Link 

https://drive.google.com/drive/folders/1uxUjmBK9a-DfvyuNEy88mOuB3WYHkw5n?usp=sharing

# Description

The implemented system represent a prototype of a federated learning system that performs digit classification. The system comprised of three main components; global model, centrelized server and clients. The global model which has Lenet architecture is trained from scratch using the first half of the MNIST training data and each client retreive the gobal model to train further using an unused portion of training data from MNIST. Centrelized server aggregates the client updates and renovates the global model while taking steps to ensure data privacy from clients ends.

# Installation

You can use requirements.txt to create a similar conda environment with all the dependencies needed for installation

# Usage

global_model.py in the global model folder is used to train the global model from scratch using the 50% of MNIST dataset where the best performed model checkpoint will be save in the beat_model.pth To execute the file simply navigate inside the global_model folder and run

python global_model.py

Take a copy of the best_model.pth and save it in the global_model_checkpoint folder under the name global_model.pth

Then execute server.py to up the server on the port 8080 of the local server and run clent.py in another command prompt to create thre clients and execute their functionality. In the script client will retrieve the global model parameter and further train for 2 epochs using a portion of MNIST training dataset and post it to the server. 

Updated global model checkpoint after each client will be saved in the global_model_update folder. To check the performance og the global model after each client update execute the eval.py. (Make sure to rename global_model_update folder before each client.py execution)

