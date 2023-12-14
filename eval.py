import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from global_model.global_model import LeNet, eval

if __name__ == "__main__":
    path = "global_model_updates"
    runs = {int("".join(f.name.split("-")[-1].split("_")[:-1])):f.name for f in os.scandir(path)}
    runs= dict(sorted(runs.items()))

    transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = LeNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    for i,file in enumerate(runs):
        model.load_state_dict(torch.load(path+"/"+runs[file]))
        model.to(device)
        val_loss, val_acc = eval(model, test_loader, criterion, device)
        print(f"After Client_{i+1} {val_loss} Accuracy {val_acc} ")