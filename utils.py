import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm

def get_device():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  return device

def print_summary(model):
  return summary(model, input_size=(1, 28, 28))

def plot_images(batch_data, batch_label):
  fig = plt.figure()
  for i in range(12):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])

def get_optimizer(model):
  optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  return optimizer

def loss_function():
  criterion = nn.CrossEntropyLoss()
  return criterion

def scheduler(optimizer):
  scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
  return scheduler

def get_train_test_data():
  # Train data transformations
  train_transforms = transforms.Compose([
      transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
      transforms.Resize((28, 28)),
      transforms.RandomRotation((-15., 15.), fill=0),
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
      ])

  # Test data transformations
  test_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
      ])
  train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
  test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
  batch_size = 512

  kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

  train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
  test_loader = torch.utils.data.DataLoader(test_data, **kwargs)  
  return train_loader, test_loader

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, criterion):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = criterion(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()

    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  return 100*correct/processed, train_loss/len(train_loader)

def test(model, device, test_loader, criterion):
  model.eval()

  test_loss = 0
  correct = 0

  with torch.no_grad():
      for batch_idx, (data, target) in enumerate(test_loader):
          data, target = data.to(device), target.to(device)

          output = model(data)
          test_loss += criterion(output, target).item()  # sum up batch loss

          correct += GetCorrectPredCount(output, target)


  test_loss /= len(test_loader.dataset)

  print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))
  return 100. * correct / len(test_loader.dataset), test_loss 

def plot_loss_acc_curves(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")


