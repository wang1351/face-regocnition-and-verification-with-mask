import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5, padding=2) #channel, number of filters, filter size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 20, 5, padding=2)
        self.fc1 = nn.Linear(20 * 4 * 4, 150)
        self.fc2 = nn.Linear(150, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 20 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ResNet18(torch.nn.Module):
  def __init__(self,n_classes=2):
    super(ResNet18, self).__init__()

    self.base_model = torch.nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-1])
    base_model_out_size = list(self.base_model.parameters())[-1].size(0)
    self.preds = torch.nn.Linear(base_model_out_size, n_classes)

  def forward(self, images):
    features = self.base_model(images).view(images.shape[0],-1)
    return self.preds(features), features #features similar