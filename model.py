import time
start_time = time.time()
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from random import sample
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

y_labels = ["Hari", "Anuram", "Aaron", "Abhay", "Jackson"]
numerical_labels = LabelEncoder().fit_transform(y_labels)
# print(numerical_labels)
# make a dictionary of labels
label_dict = dict(zip(y_labels, numerical_labels))
# reversed dictionary
compare_dict = dict((v,k) for k,v in label_dict.items())
print(compare_dict , "compare_dict")
print(label_dict)

class FaceDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        contents = os.listdir(data_dir)
        self.images = [f for f in contents if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.JPG')]  
        self.images.sort()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((200,200)) ,
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.data_dir, image_name)
        name = image_name.split('_')[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = self.transform(image)
        label = label_dict[name]
        label = torch.tensor(label)
        sample = {
            "data": image,
            "label": label 
        }
        return sample
    
train_data_dir = 'train_facedataset'
train_dataset = FaceDataset(train_data_dir)
print(len(train_dataset))

test_data_dir = 'test_facedataset'
test_dataset = FaceDataset(test_data_dir)
print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True)


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 25 * 25, 256),  # Adjusted input size
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 5) , # Adjusted output size
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        # print(out.shape , "out layer 3")
        out = out.view(out.size(0), -1) 
        out = self.fc(out)
        return out


model = CNNModel()
epochs = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(epochs):
    model.train()
    
    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    for i, sample in enumerate(train_loader):
        inputs = sample['data']
        labels = sample['label']
        print(inputs.shape , labels.shape , "batch" , i , "epoch" , epoch)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f"Epoch {epoch}, loss: {running_loss}, accuracy: {accuracy}")
   
    model_filename = f"model_epoch_{epoch}.pth"
    torch.save(model.state_dict(), model_filename)
    print(f"Saved model at epoch {epoch} as {model_filename}")


# torch.save(model.state_dict(), 'model.pth')

end_time = time.time()  # End timing
training_time = end_time - start_time  # Calculate training time
print('Finished Training')
print("model training time = " , training_time)

print("Model trained and saved successfully")

#code for model summary
from torchsummary import summary
summary(model, (1, 200, 200))




