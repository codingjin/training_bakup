import os
import math
import torch
import numpy as np
import codecs
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from torchvision import models

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("No GPU available; training on CPU")
custom_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
])
class CustomDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_list = []
        reader = codecs.open(label_file, 'r', 'utf-8')
        lines = reader.readlines()
        for line in lines:
            parts = line.split('\t')
            image_file = parts[0]
            label = int(parts[1])
            image_file_path = os.path.join(data_dir, image_file)
            image = Image.open(image_file_path)
            if image.mode != 'RGB':
                continue
            self.data_list.append((image_file_path, label))
        self.data_list = self.data_list * 10

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label = self.data_list[index]
        image = Image.open(image_path)
        data = custom_transform(image)
        return data, label
batch_size = 32
trainset = CustomDataset('/home/jin/imagenet/train', '/home/jin/imagenet/train_label.txt')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = CustomDataset('/home/jin/imagenet/test', '/home/jin/imagenet/test_label.txt')
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

model = models.resnet101(pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
best_accuracy = 0
for epoch in range(1000):  # Change the number of epochs as needed
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 mini-batches
            print(f'[Epoch {epoch + 1}, Mini-Batch {i + 1}] Loss: {running_loss / 10:.3f}')
            running_loss = 0.0
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()
        # initialize a list to store our predictions
        accuracy = 0
        predictions = []
        true_labels = []
        for inputs, labels in testloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # Get the predicted class labels
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        correct = (predictions == true_labels).sum()
        total = len(predictions)
        accuracy = correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), f'{best_accuracy}.pt')
