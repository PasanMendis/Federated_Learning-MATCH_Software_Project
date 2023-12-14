import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import ssl
import logging

# Global model (AlexNet)
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


criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

def train(model, train_loader, optimizer, criterion, device):
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


def eval(model, test_loader, criterion, device):
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
  

def main():

    logging.basicConfig(filename="training.log",
                    format='%(asctime)s %(message)s',
                    filemode='w',
                    level=logging.INFO)

    logger = logging.getLogger()

    writer = SummaryWriter(comment="ERM")
    # Initialize server model
    model = LeNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Load MNIST dataset
    mnist_mean,mnist_std = (0.1307,),(0.3081,)

    train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mnist_mean, mnist_std),
            ])
    test_mnist_transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize(mnist_mean, mnist_std),
                ])

    train_dataset = datasets.MNIST(
            root='./data', train=True, transform=train_transform, download=True)
    # Only taking a portion of data for demonstration
    train_dataset.data = train_dataset.data[:30000] 
    train_dataset.targets = train_dataset.targets[:30000]

    test_dataset = datasets.MNIST(
                root='./data', train=False, transform=test_mnist_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Train the global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    top_val =0
    # Define Epoch count
    epoch_count = 7
    for epoch in range(epoch_count):  # Number of federated learning rounds
        
        train_loss,train_acc = train(model, train_loader, optimizer, criterion, device)
        
        print(f"Epoch: {epoch+1}/{epoch_count} Train_loss {train_loss} Train_acc {train_acc} ")
        logger.info(f"Epoch: {epoch+1}/{epoch_count} Train_loss {train_loss} Train_acc {train_acc} ")
        writer.add_scalar("Loss/Train",train_loss,epoch)
        writer.add_scalar("Accuracy/Train",train_acc,epoch)

        val_loss, val_acc = eval(model, test_loader, criterion, device)
        print(f"Validation++++++ loss {val_loss} Accuracy {val_acc} ")
        logger.info(f"Validation++++++ loss {val_loss} Accuracy {val_acc} ")
        writer.add_scalar("Loss/Val",val_loss,epoch)
        writer.add_scalar("Accuracy/Val",val_acc,epoch)

        if top_val<(val_acc*100):
            print("Best Model Upto Now ________________")
            logger.info("Best Model Upto Now ________________")
            top_val=(val_acc*100)
            torch.save(model.state_dict(), "best_model.pth")


    # Save the trained global model
    torch.save(model.state_dict(), "global_model.pth")
    writer.close()

if __name__ == "__main__":
    main()