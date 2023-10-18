import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import sys

torch.manual_seed(17)

class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, in_shape=None):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        if in_shape == None:
            print("in_shape could not be none!")
            sys.exit(1)
        if stride == 2:
            self.in_shape = [int(in_shape[0]/2), int(in_shape[1]/2)]
        else:
            self.in_shape = in_shape
        self.ln1 = nn.LayerNorm([out_channels] + self.in_shape)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        self.ln2 = nn.LayerNorm([out_channels] + self.in_shape)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.conv2(x)
        #x = self.bn2(x)
        x = self.ln2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    

class ResNet18(nn.Module):
    
    def __init__(self, input_shape=[32,32], image_channels=3, num_classes=10):
        
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        #self.bn1 = nn.BatchNorm2d(64)
        self.in_shape = [int(input_shape[0]/2), int(input_shape[1]/2)]
        self.ln1 = nn.LayerNorm([self.in_channels] + self.in_shape)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_shape = [int(self.in_shape[0]/2), int(self.in_shape[1]/2)]
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1, in_shape=self.in_shape)

        self.layer2 = self.__make_layer(64, 128, stride=2, in_shape=self.in_shape)
        self.in_shape[0], self.in_shape[1] = int(self.in_shape[0]/2), int(self.in_shape[1]/2)

        self.layer3 = self.__make_layer(128, 256, stride=2, in_shape=self.in_shape)
        self.in_shape[0], self.in_shape[1] = int(self.in_shape[0]/2), int(self.in_shape[1]/2)
        
        self.layer4 = self.__make_layer(256, 512, stride=2, in_shape=self.in_shape)
        self.in_shape[0], self.in_shape[1] = int(self.in_shape[0]/2), int(self.in_shape[1]/2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride, in_shape=None):
        
        identity_downsample = None
        if stride != 1:
            if (in_shape == None):
                print("Invalid in_shape parameter!")
                sys.exit(1)
            
            identity_downsample = self.identity_downsample(in_channels, out_channels, in_shape)
            return nn.Sequential(
                Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride, in_shape=in_shape),
                Block(out_channels, out_channels, in_shape=[int(in_shape[0]/2), int(in_shape[1]/2)])
            )
        else:
            return nn.Sequential(
                Block(in_channels, out_channels, in_shape=in_shape),
                Block(out_channels, out_channels, in_shape=in_shape)
            )
        
    def forward(self, x):
        
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x 
    
    def identity_downsample(self, in_channels, out_channels, in_shape):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1), 
            #nn.BatchNorm2d(out_channels)
            nn.LayerNorm([out_channels] + [int(in_shape[0]/2), int(in_shape[1]/2)])
        )    

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = ResNet18().to(device)
#print(model)


