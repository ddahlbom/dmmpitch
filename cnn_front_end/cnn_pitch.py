import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/home/dahlbom/research/dmm_pitch/data_gen/')
import pickle
import glob


################################################################################
# Import and transform data
################################################################################
file_prefix = "jsb_data"
file_path   = "/home/dahlbom/research/dmm_pitch/data_gen/"

class auditoryDataset(Dataset):
    def __init__(self, file_path, file_prefix, transform=None):
        glob_path = file_path + file_prefix + "*.bin"
        files = glob.glob(glob_path)
        x = []  # auditory images 
        y = []  # labels
        for k, f in enumerate(files):
            data = pickle.load(open(f, "rb"))
            y.append(data[0])
            x.append(data[1])
            if k == 0: 
                break
        self.images = np.concatenate(x)
        self.notes  = np.concatenate(y)
        print(self.images.shape)
        print(self.notes.shape)
        assert self.images.shape[0] == self.notes.shape[0]
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        note_slice = self.notes[idx]
        sample = {'image' : image, 'notes' : note_slice}
        if self.transform:
            sample = self.transform(sample)
        return sample

class Renorm(object):

    def __call__(self, sample):
        image, notes = sample['image'], sample['notes']
        image /= np.max(image)  #normalize between 0 and 1
        image -= 0.5    # center at 0
        
        return {'image': image, 'notes': notes}

class ToTensor(object):

    def __call__(self, sample):
        image, notes = sample['image'], sample['notes']
        image = image.transpose()
        image = image.reshape((1,image.shape[0],image.shape[1]))
        return {'image': torch.from_numpy(image),
                'notes': torch.from_numpy(notes)}


bach_dataset = auditoryDataset(file_path,
                               file_prefix,
                               transform=transforms.Compose([Renorm(),
                                                             ToTensor()]))
fig = plt.figure()
ax = []
# for i in range(len(bach_dataset)):
offset = 100
for i in range(offset,offset+4):
    sample = bach_dataset[i]
    notes = sample['notes']
    note_idcs = np.where(notes==1)
    image = sample['image']
    ax.append(fig.add_subplot(2,2,i-offset+1))
    ax[i-offset].imshow(image[0].transpose(1,0), aspect='auto' )
    ax[i-offset].set_title(note_idcs)
    
plt.show()    


'''
################################################################################
# Network definition
################################################################################

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input channel, six output channels, 5x5 kernels
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))    # same as (2,2)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, "\n")
net.to(device)


################################################################################
# Loss Function and Optimizer
################################################################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


################################################################################
# Training
################################################################################
num_epochs = 2
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')
'''
