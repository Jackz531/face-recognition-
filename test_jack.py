import cv2
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.calibration import LabelEncoder
from torchvision.transforms import functional as F
import numpy as np
import time
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
 
y_labels = ["Hari", "Anuram", "Aaron", "Abhay","Jackson" ]
numerical_labels = LabelEncoder().fit_transform(y_labels)
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
        # print(name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = self.transform(image)
        # print(image.shape)
        label = label_dict[name]
        label = torch.tensor(label)
        # print(label)
        sample = {
            "data": image,
            "label": label 
        }
        return sample

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
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Function to test the model and draw bounding boxes using Haar Cascade
def test_model_with_haar_and_plot(model, test_loader, compare_dict):
    model.eval()
    correct = 0
    total = 0
    results = []
    start_time = time.time()

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            inputs = sample['data']
            labels = sample['label']
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Process each image in the batch
            for j in range(inputs.size(0)):
                # Convert tensor to PIL Image for Haar Cascade
                pil_image = F.to_pil_image(inputs[j].cpu())
                open_cv_image = np.array(pil_image)

                # Check if the image is grayscale and convert to BGR if necessary
                if len(open_cv_image.shape) == 2:
                    # Convert grayscale image to color (3 channels)
                    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)

                # Convert BGR to RGB
                open_cv_image = open_cv_image[:, :, ::-1].copy()

                # Detect faces in the image
                faces = face_cascade.detectMultiScale(open_cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Draw bounding box and label around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(open_cv_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    label = compare_dict[predicted[j].item()]
                    cv2.putText(open_cv_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

                # Convert RGB to BGR for OpenCV
                open_cv_image = open_cv_image[:, :, ::-1].copy()

                # Save the image with bounding boxes
                cv2.imwrite(f'test_output_{i}_{j}.jpg', open_cv_image)

                # Append results for CSV
                results.append([i*inputs.size(0) + j + 1, compare_dict[predicted[j].item()], compare_dict[labels[j].item()]])

                # Plot the image with bounding boxes and labels
                plt.figure(figsize=(10, 10))
                plt.imshow(open_cv_image)
                plt.title(f"Predicted: {label}  Actual : {compare_dict[labels[j].item()]}")

                plt.axis('off')
                plt.show()

    # Calculate accuracy
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
    print(f"Tested in {time.time() - start_time} seconds")

    # Create a DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=['Serial No', 'Predicted Label', 'Actual Label'])
    results_df.to_csv('test_results.csv', index=False)

    return accuracy

# Load the model
model = CNNModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Load the test dataset
test_data_dir = 'test_facedataset'
test_dataset = FaceDataset(test_data_dir)
test_loader = DataLoader(test_dataset, batch_size=3, shuffle=True)

# Call the testing function
test_model_with_haar_and_plot(model, test_loader, compare_dict)