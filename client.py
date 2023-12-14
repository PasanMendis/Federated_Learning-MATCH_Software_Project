from http import client
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from collections import OrderedDict

class LeNet(nn.Module):
  def __init__(self, num_classes=10, **kwargs):
      super(LeNet, self).__init__()
      self.features = nn.Sequential(
          nn.Conv2d(1, 64, 5),
          nn.ReLU(True),
          nn.MaxPool2d(2, 2),
          nn.Conv2d(64, 128, 5),
          nn.ReLU(True),
          nn.MaxPool2d(2, 2)
      )
      self.classifier = nn.Sequential(
          nn.Linear(128 * 5 * 5, 1024),
          nn.ReLU(True),
          nn.Linear(1024, 1024),
          nn.ReLU(True),
          nn.Linear(1024, num_classes)
      )

  def forward(self, x):
      x = self.features(x)
      x = self.classifier(x.view(x.size(0), -1))
      return x

class Client:
    def __init__(self, train_partion,epoch_count):
        self.train_partion = train_partion
        self.epoch_count = epoch_count

    def train(self,model, train_loader, optimizer, criterion, device):
        model.train()
        train_loss =0
        train_acc =0
        train_n=0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * target.size(0)
            train_acc += (output.max(1)[1] == target).sum().item()
            train_n += target.size(0)

        return train_loss/train_n, train_acc/train_n
    
    def eval(self, model, test_loader, criterion, device):
        model.eval()
        test_loss =0
        test_acc =0
        test_n=0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item() * target.size(0)
            test_acc += (output.max(1)[1] == target).sum().item()
            test_n += target.size(0)
        return test_loss/test_n, test_acc/test_n
    
    def get_loaders(self):
        transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        partition_begin=30000+(self.train_partion-1)*10000
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        train_dataset.data, train_dataset.targets = train_dataset.data[partition_begin:partition_begin+10000], train_dataset.targets[partition_begin:partition_begin+10000]
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

        return train_loader, test_loader
    
    def update_local_model(self):
        local_model = LeNet()
        # Get the global model from the server
        global_model_url = 'https://127.0.0.1:8080/get_global_model'
        auth = ('user', 'password')
        response = requests.get(global_model_url, auth=auth, verify='localhost.crt')
        params = OrderedDict({k: torch.Tensor(response.json()[k]) for k in response.json()})
        local_model.load_state_dict(params)

        return local_model
    
    def send_local_model(self, model):
        model_state = model.state_dict()
        model_state = OrderedDict({k: v.detach().cpu().tolist() for k, v in model_state.items()})
        
        model_grads = [param.grad.clone().detach().cpu().tolist() for param in model.parameters()]

        model_info = {"model_state":model_state, "grad":model_grads}

        update_global_model_url = 'https://127.0.0.1:8080/update_global_model'
        auth = ('user', 'password')
        response = requests.post(update_global_model_url, json=json.dumps(model_info), auth=auth, verify='localhost.crt')

        if response.status_code == 200:
            print("Global model updated successfully")
        else:
            print("Failed to update global model")

    def process_client(self):
        # get client specific data
        train_loader, test_loader = self.get_loaders()

        # get the global model from the server to train the local model
        local_model = self.update_local_model()

        optimizer = optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Train the local model
        best_model=None
        top_val=0
        local_model.to(device)

        for epoch in range(self.epoch_count):  # Number of local training rounds

            train_loss,train_acc = self.train(local_model, train_loader, optimizer, criterion, device)
            print(f"Epoch: {epoch+1}/{self.epoch_count} Train_loss {train_loss} Train_acc {train_acc} ")

            val_loss, val_acc = self.eval(local_model, test_loader, criterion, device)
            print(f"Validation++++++ loss {val_loss} Accuracy {val_acc} ")
        
            if top_val<(val_acc*100):
                print("Best Model Upto Now ________________")
                top_val=(val_acc*100)
                best_model = local_model

        print("ALL trained _____________")

        # send the local model to server
        self.send_local_model(best_model)


if __name__ == "__main__":

    # create clients
    for p in range(1,4):
        print(f"Processes Client_{p} ***********************************")
        clients = Client(p,2)
        clients.process_client()

    print("Processed ALL Clients #############################")

