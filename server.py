from flask import Flask, request
from flask_httpauth import HTTPBasicAuth
import ssl
import torch
from global_model.global_model import LeNet
import json
from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime
import os


app = Flask(__name__)
auth = HTTPBasicAuth()

context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.load_cert_chain('localhost.crt', 'localhost.key')

@auth.verify_password
def verify_password(username, password):
    return username == 'user' and password == 'password'

@app.route('/get_global_model', methods=['GET'])
@auth.login_required
def get_global_model():
    global global_model_path

    model_params = torch.load(global_model_path)
    model_params = OrderedDict({k: model_params[k].detach().cpu().tolist() for k in model_params})
    return json.dumps(model_params)

@app.route('/update_global_model', methods=['POST'])
@auth.login_required
def update_global_model():
    global global_model_path

    #Load the global model 
    global_model=LeNet()
    global_model.load_state_dict(torch.load(global_model_path))
    
    
    #Load the client model
    client_model_info = json.loads(request.get_json())
    client_model_state_dict = client_model_info["model_state"]
    client_model_state = OrderedDict({k: torch.Tensor(client_model_state_dict[k]) for k in client_model_state_dict})
    client_gradients = [torch.Tensor(v) for v in client_model_info["grad"]]

    # Anonymize data for privacy - Adding random noise for anonymization
    for k in client_model_state: client_model_state[k] = client_model_state[k]+0.0002*torch.randn_like(client_model_state[k])

    # Differential privacy parameter 
    epsilon = 10.0  
    noise_multiplier = 0.00001
    clip_value = 0.005  

    # Add noise to gradients for differential privacy
    for param, gradient in zip(global_model.parameters(), client_gradients):
        noise = torch.tensor(clip_value, dtype=param.dtype).normal_() * noise_multiplier
        gradient.add_(noise / epsilon)

    # Update the global model with the differentially private gradients
    global_optimizer=optim.SGD(global_model.parameters(), lr=0.01, momentum=0.9)
    global_optimizer.zero_grad()
    global_model.load_state_dict(client_model_state)
    global_optimizer.step()

    # Save the updated global model
    current_dateTime = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

    if not os.path.exists("global_model_updates"):
        os.mkdir("global_model_updates")

    torch.save(global_model.state_dict(), "global_model_updates/global_model_with_client_"+current_dateTime+".pth")
    torch.save(global_model.state_dict(), global_model_path)
    return 'Update received', 200


if __name__ == '__main__':
    global_model_path = "global_model_checkpoint/global_model.pth"
    app.run(ssl_context=context, port=8080)
