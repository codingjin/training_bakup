import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision._internally_replaced_utils import load_state_dict_from_url

torch.manual_seed(17)

class Block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1, use_layer_norm=False):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        if use_layer_norm:
            self.bn1 = None #TODO
        else:
            self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    

class ResNet18(nn.Module):
    
    def __init__(self, image_channels=3, num_classes=10, pretrained_weights=True, state_dict=None, use_layer_norm=False):
        
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        if use_layer_norm:
            self.bn1 = None
            #TODO
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1, use_layer_norm=use_layer_norm)
        self.layer2 = self.__make_layer(64, 128, stride=2, use_layer_norm=use_layer_norm)
        self.layer3 = self.__make_layer(128, 256, stride=2, use_layer_norm=use_layer_norm)
        self.layer4 = self.__make_layer(256, 512, stride=2, use_layer_norm=use_layer_norm)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        if pretrained_weights:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
        if state_dict is not None:
            self._load_state_dict(state_dict, use_layer_norm=False)
    
    def _map_to_class_layers_layer_norm(self):
        return {
            "conv1.weight": self.conv1,
            "layer1.0.conv1.weight": self.layer1[0].conv1,
            "layer1.0.conv2.weight": self.layer1[0].conv2,
            "layer1.1.conv1.weight": self.layer1[1].conv1,
            "layer1.1.conv2.weight": self.layer1[1].conv2,
            "layer2.0.conv1.weight": self.layer2[0].conv1,
            "layer2.0.conv2.weight": self.layer2[0].conv2,
            "layer2.0.downsample.0.weight": self.layer2[0].identity_downsample._modules['0'],
            "layer2.1.conv1.weight": self.layer2[1].conv1,
            "layer2.1.conv2.weight": self.layer2[1].conv2,
            "layer3.0.conv1.weight": self.layer3[0].conv1,
            "layer3.0.conv2.weight": self.layer3[0].conv2,
            "layer3.0.downsample.0.weight": self.layer3[0].identity_downsample._modules['0'],
            "layer3.1.conv1.weight": self.layer3[1].conv1,
            "layer3.1.conv2.weight": self.layer3[1].conv2,
            "layer4.0.conv1.weight": self.layer4[0].conv1,
            "layer4.0.conv2.weight": self.layer4[0].conv2,
            "layer4.0.downsample.0.weight": self.layer4[0].identity_downsample._modules['0'],
            "layer4.1.conv1.weight": self.layer4[1].conv1,
            "layer4.1.conv2.weight": self.layer4[1].conv2,
            # fc.weight: ,
            # fc.bias: ,
        }
        
    def _map_to_class_layers_batchnorm2d(self):
        return {
            "conv1.weight": self.conv1,
            "bn1.running_mean": self.bn1,
            "bn1.running_var": self.bn1,
            "bn1.weight": self.bn1,
            "bn1.bias": self.bn1,
            "layer1.0.conv1.weight": self.layer1[0].conv1,
            "layer1.0.bn1.running_mean": self.layer1[0].bn1,
            "layer1.0.bn1.running_var": self.layer1[0].bn1,
            "layer1.0.bn1.weight": self.layer1[0].bn1,
            "layer1.0.bn1.bias": self.layer1[0].bn1,
            "layer1.0.conv2.weight": self.layer1[0].conv2,
            "layer1.0.bn2.running_mean": self.layer1[0].bn2,
            "layer1.0.bn2.running_var": self.layer1[0].bn2,
            "layer1.0.bn2.weight": self.layer1[0].bn2,
            "layer1.0.bn2.bias": self.layer1[0].bn2,
            "layer1.1.conv1.weight": self.layer1[1].conv1,
            "layer1.1.bn1.running_mean": self.layer1[1].bn1,
            "layer1.1.bn1.running_var": self.layer1[1].bn1,
            "layer1.1.bn1.weight": self.layer1[1].bn1,
            "layer1.1.bn1.bias": self.layer1[1].bn1,
            "layer1.1.conv2.weight": self.layer1[1].conv2,
            "layer1.1.bn2.running_mean": self.layer1[1].bn2,
            "layer1.1.bn2.running_var": self.layer1[1].bn2,
            "layer1.1.bn2.weight": self.layer1[1].bn2,
            "layer1.1.bn2.bias": self.layer1[1].bn2,
            "layer2.0.conv1.weight": self.layer2[0].conv1,
            "layer2.0.bn1.running_mean": self.layer2[0].bn1,
            "layer2.0.bn1.running_var": self.layer2[0].bn1,
            "layer2.0.bn1.weight": self.layer2[0].bn1,
            "layer2.0.bn1.bias": self.layer2[0].bn1,
            "layer2.0.conv2.weight": self.layer2[0].conv2,
            "layer2.0.bn2.running_mean": self.layer2[0].bn2,
            "layer2.0.bn2.running_var": self.layer2[0].bn2,
            "layer2.0.bn2.weight": self.layer2[0].bn2,
            "layer2.0.bn2.bias": self.layer2[0].bn2,
            "layer2.0.downsample.0.weight": self.layer2[0].identity_downsample._modules['0'],
            "layer2.0.downsample.1.running_mean": self.layer2[0].identity_downsample._modules['1'],
            "layer2.0.downsample.1.running_var": self.layer2[0].identity_downsample._modules['1'],
            "layer2.0.downsample.1.weight": self.layer2[0].identity_downsample._modules['1'],
            "layer2.0.downsample.1.bias": self.layer2[0].identity_downsample._modules['1'],
            "layer2.1.conv1.weight": self.layer2[1].conv1,
            "layer2.1.bn1.running_mean": self.layer2[1].bn1,
            "layer2.1.bn1.running_var": self.layer2[1].bn1,
            "layer2.1.bn1.weight": self.layer2[1].bn1,
            "layer2.1.bn1.bias": self.layer2[1].bn1,
            "layer2.1.conv2.weight": self.layer2[1].conv2,
            "layer2.1.bn2.running_mean": self.layer2[1].bn2,
            "layer2.1.bn2.running_var": self.layer2[1].bn2,
            "layer2.1.bn2.weight": self.layer2[1].bn2,
            "layer2.1.bn2.bias": self.layer2[1].bn2,
            "layer3.0.conv1.weight": self.layer3[0].conv1,
            "layer3.0.bn1.running_mean": self.layer3[0].bn1,
            "layer3.0.bn1.running_var": self.layer3[0].bn1,
            "layer3.0.bn1.weight": self.layer3[0].bn1,
            "layer3.0.bn1.bias": self.layer3[0].bn1,
            "layer3.0.conv2.weight": self.layer3[0].conv2,
            "layer3.0.bn2.running_mean": self.layer3[0].bn2,
            "layer3.0.bn2.running_var": self.layer3[0].bn2,
            "layer3.0.bn2.weight": self.layer3[0].bn2,
            "layer3.0.bn2.bias": self.layer3[0].bn2,
            "layer3.0.downsample.0.weight": self.layer3[0].identity_downsample._modules['0'],
            "layer3.0.downsample.1.running_mean": self.layer3[0].identity_downsample._modules['1'],
            "layer3.0.downsample.1.running_var": self.layer3[0].identity_downsample._modules['1'],
            "layer3.0.downsample.1.weight": self.layer3[0].identity_downsample._modules['1'],
            "layer3.0.downsample.1.bias": self.layer3[0].identity_downsample._modules['1'],
            "layer3.1.conv1.weight": self.layer3[1].conv1,
            "layer3.1.bn1.running_mean": self.layer3[1].bn1,
            "layer3.1.bn1.running_var": self.layer3[1].bn1,
            "layer3.1.bn1.weight": self.layer3[1].bn1,
            "layer3.1.bn1.bias": self.layer3[1].bn1,
            "layer3.1.conv2.weight": self.layer3[1].conv2,
            "layer3.1.bn2.running_mean": self.layer3[1].bn2,
            "layer3.1.bn2.running_var": self.layer3[1].bn2,
            "layer3.1.bn2.weight": self.layer3[1].bn2,
            "layer3.1.bn2.bias": self.layer3[1].bn2,
            "layer4.0.conv1.weight": self.layer4[0].conv1,
            "layer4.0.bn1.running_mean": self.layer4[0].bn1,
            "layer4.0.bn1.running_var": self.layer4[0].bn1,
            "layer4.0.bn1.weight": self.layer4[0].bn1,
            "layer4.0.bn1.bias": self.layer4[0].bn1,
            "layer4.0.conv2.weight": self.layer4[0].conv2,
            "layer4.0.bn2.running_mean": self.layer4[0].bn2,
            "layer4.0.bn2.running_var": self.layer4[0].bn2,
            "layer4.0.bn2.weight": self.layer4[0].bn2,
            "layer4.0.bn2.bias": self.layer4[0].bn2,
            "layer4.0.downsample.0.weight": self.layer4[0].identity_downsample._modules['0'],
            "layer4.0.downsample.1.running_mean": self.layer4[0].identity_downsample._modules['1'],
            "layer4.0.downsample.1.running_var": self.layer4[0].identity_downsample._modules['1'],
            "layer4.0.downsample.1.weight": self.layer4[0].identity_downsample._modules['1'],
            "layer4.0.downsample.1.bias": self.layer4[0].identity_downsample._modules['1'],
            "layer4.1.conv1.weight": self.layer4[1].conv1,
            "layer4.1.bn1.running_mean": self.layer4[1].bn1,
            "layer4.1.bn1.running_var": self.layer4[1].bn1,
            "layer4.1.bn1.weight": self.layer4[1].bn1,
            "layer4.1.bn1.bias": self.layer4[1].bn1,
            "layer4.1.conv2.weight": self.layer4[1].conv2,
            "layer4.1.conv1.weight": self.layer4[1].conv1,
            "layer4.1.bn2.running_mean": self.layer4[1].bn2,
            "layer4.1.bn2.running_var": self.layer4[1].bn2,
            "layer4.1.bn2.weight": self.layer4[1].bn2,
            "layer4.1.bn2.bias": self.layer4[1].bn2,
            # fc.weight: ,
            # fc.bias: ,
        }

    def _load_state_dict(self, state_dict, use_layer_norm=False) -> None:
        if use_layer_norm:
            map = self._map_to_class_layers_layer_norm()
        else:
            map = self._map_to_class_layers_batchnorm2d()
        for key in state_dict.keys():
            if key in map:
                layer = map[key]
                if isinstance(layer, nn.Conv2d):
                    layer.weight = nn.Parameter(state_dict[key])
                if isinstance(layer, nn.BatchNorm2d):
                    if 'running_mean' in key:
                        layer.running_mean = nn.Parameter(state_dict[key])
                    elif 'running_var' in key:
                        layer.running_var = nn.Parameter(state_dict[key])
                    elif 'weight' in key:
                        layer.weight = nn.Parameter(state_dict[key])
                    elif 'bias' in key:
                        layer.bias = nn.Parameter(state_dict[key])
            else:
                print(key, "unused from state_dict")
                    

    def __make_layer(self, in_channels, out_channels, stride, use_layer_norm=False):
        
        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels, stride, use_layer_norm)
            
        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride, use_layer_norm=use_layer_norm), 
            Block(out_channels, out_channels, use_layer_norm=use_layer_norm)
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
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
    
    def identity_downsample(self, in_channels, out_channels, stride, use_layer_norm=False):
        if use_layer_norm:
            return None #TODO
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(out_channels)
            )    

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = ResNet18().to(device)
#print(model)


