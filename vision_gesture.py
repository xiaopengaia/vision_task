import torch
import os
import cv2
import numpy as np
import os
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from dataset import * 
from torchvision.models.resnet import resnet18
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')
import torch
from PIL import Image
from dataset import *
from torchvision.models.resnet import resnet18

import torch
from dataset import *
from tqdm import tqdm
from torchvision.models.resnet import resnet18
import random
import cv2
import numpy as np
from test import *
from torchvision.models.resnet import resnet18



"""
deal datasets
"""

labels = "./image/test/labels.txt"

with open(labels, "w") as f:
    for i in range(0, 10):
        data_path = f"./image/test/{i}"
        for index, image_name in enumerate(os.listdir(data_path)):
            image_path = os.path.join(data_path, image_name)
            f.write(f"{image_path} {i}\n")

"""
get mean and std value
"""
def get_mean_std(root):
    image_path, label_path = [], []
    with open(root, "r") as f:
        file_names = f.readlines()
        for file_name in file_names:
            image_name = file_name.split()[0]
            label_name = int(file_name.split()[1])
            image_path.append(image_name)
            label_path.append([label_name])
    nimages, mean, std = 0, 0, 0
    for image_name in image_path:
        nimages += 1
        image = cv2.imread(image_name)
        # image = image / 255
        image = image.reshape(-1, 3)
        mean += image.mean(0)
        std += image.std(0)
    mean /= nimages
    std /= nimages
    return mean, std

"""
custom Dataset
"""
class ImageData(Dataset):
    def __init__(self, root, training=True):
        super().__init__()
        self.images, self.labels = [], []
        with open(root, "r") as f:
            file_names = f.readlines()
            for file_name in file_names:
                image_name = file_name.split()[0]
                label_name = int(file_name.split()[1])
                self.images.append(image_name)
                self.labels.append(label_name)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = torch.tensor(self.labels[index])
        images = self.images[index]
        transfor = transforms.Compose([
            lambda image : Image.open(image).convert("RGB"),
            # lambda image : Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            transforms.ToTensor(),
            transforms.Normalize([0.506, 0.521, 0.535], [0.098, 0.096, 0.063])])
        image = transfor(images)
        return image, label


"""
train
"""

device = "cuda:0"
data_path = "./image/train/labels.txt"
datasets = DataLoader(ImageData(data_path), batch_size=20, shuffle=True, num_workers=8)
net = resnet18().to(device)
net.fc = nn.Linear(512, 10).to(device)
crossentropy = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

net.train()
print("trian start")
for epoch in range(50):
    for image, label in datasets:
        image = Variable(image).to(device)
        label = Variable(label).to(device)
        
        output = net(image)
        _, predict = torch.max(output, axis=1)

        optimizer.zero_grad()
        loss = crossentropy(output, label)

        loss.backward()
        optimizer.step()
    print(f"epoch = {epoch}\tloss = {loss}")

torch.save(net, "net18_2.pt")


"""
test
"""

for image in datasets:
    output = net(image.cuda())
    result = torch.argmax(output, axis=1)
    if int(result[0]) == 0:
        return 10
    else:
        return int(result[0])


"""
acc
"""
data_path = "./image/test/labels.txt"
dataloader = DataLoader(ImageData(data_path), batch_size=20, shuffle=True, num_workers=8)

net = torch.load("./net18.pt").cuda()
net.eval()

with torch.no_grad():
    acc_net, total = 0, 0
    for i, (images, labels) in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        images = images.cuda()
        labels = labels.cuda()

        result_net = net(images)

        result_net_prob = torch.argmax(result_net, axis=1)

        acc_net += (result_net_prob == labels).sum()
        total += len(labels)

    print("acc_net = ", acc_net / total)


"""
app
"""

def get_func():
    while True:
        num = random.randint(2, 10)

        add_num1 = []
        add_num2 = []

        for i in range(10):
            for j in range(10):
                if i+j == num:
                    add_num1.append(i)
                    add_num2.append(j)

        if len(add_num1) == 1:
            add_index = 0
        else:
            add_index = random.randint(0, len(add_num1)-1)
        
        result = add_num1[add_index] + add_num2[add_index]

        return f"{add_num1[add_index]} + {add_num2[add_index]} = ", result


def draw_func(image):
    text, result = get_func()
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, text, (90, 350), font, 3, (255, 255, 255), 3)
    return image, result


net = torch.load("./net18.pt").cuda()
net.eval()
font = cv2.FONT_HERSHEY_SIMPLEX 
vid = cv2.VideoCapture(0)
count = 0
while True:
    image_bk = np.zeros((600, 1000, 3), dtype=np.uint8)
    image_bk, result = draw_func(image_bk)
    k = 0
    while True:
        count += 1
        return_value, image = vid.read()
        image = cv2.resize(image, (224, 224))
        image_bk[200:424, 620:844,:] = image
        if count % 50 == 0:
            num1 = gtest(net, image)
            if num1 == result:
                image_an = cv2.putText(image_bk, "YOU ARE RIGHT", (50, 150), font, 1.5, (255, 255, 255), 2)
                image_an = cv2.putText(image_an, str(num1), (520, 350), font, 3, (0, 255, 255), 3)
                cv2.imshow("image", image_an)
                cv2.waitKey(2000)
                break
            else:
                image_bk = cv2.putText(image_bk, "YOU ARE WRONG", (50, 150), font, 1.5, (255, 255, 255), 2)
                image_bk = cv2.putText(image_bk, str(num1), (520, 350), font, 3, (0, 255, 255), 3)
                cv2.imshow("image", image_bk)
                cv2.waitKey(2000)
                break
        cv2.imshow("image", image_bk)
        k = cv2.waitKey(100)
    if k == 27:
        break

