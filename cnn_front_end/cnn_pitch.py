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
# Classes for data set processing
################################################################################

class auditoryDataset(Dataset):
    def __init__(self, file_path, file_prefix, transform=None):
        glob_path = file_path + file_prefix + "*.bin"
        print("path:", glob_path)
        files = glob.glob(glob_path)
        print(files)
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


################################################################################
# Network definition
################################################################################

class Net(nn.Module):

    @staticmethod
    def calc_dim_output(img_dim, filt_dim, stride, pad_dim):
        return (img_dim - filt_dim + 2*pad_dim)/stride + 1

    def __init__(self, img_size, drop_p=0.0):
        super(Net, self).__init__()

        # size parameters
        h_dim = img_size[0]
        w_dim = img_size[1]
        
        # First convolutional layer
        f1_h_dim = 25 
        f1_w_dim = 5
        padding1 = 0
        stride1 = 1
        self.conv1 = nn.Conv2d(1, 32, (f1_h_dim, f1_w_dim), stride=stride1,
                               padding=padding1)
        h_dim = self.calc_dim_output(h_dim, f1_h_dim, stride1, padding1)
        w_dim = self.calc_dim_output(w_dim, f1_w_dim, stride1, padding1)
        print("after convo1: ", h_dim, w_dim)
        h_dim = int(h_dim)
        w_dim = int(w_dim)

        # First pooling layer
        pool1_h_dim = 4
        pool1_w_dim = 4
        self.pool1 = nn.MaxPool2d(pool1_h_dim, pool1_w_dim)
        h_dim = h_dim/pool1_h_dim
        w_dim = w_dim/pool1_w_dim
        print("after pool1: ", h_dim, w_dim)
        h_dim = int(h_dim)
        w_dim = int(w_dim)

        # Second convolutional layer
        f2_h_dim = 15
        f2_w_dim = 3 
        padding2 = 0
        stride2 = 1
        self.conv2 = nn.Conv2d(32, 16, (f2_h_dim, f2_w_dim))
        h_dim = self.calc_dim_output(h_dim, f2_h_dim, stride2, padding2)
        w_dim = self.calc_dim_output(w_dim, f2_w_dim, stride2, padding2)
        print("after conv2: ", h_dim, w_dim)
        h_dim = int(h_dim)
        w_dim = int(w_dim)

        # Second pooling layer
        pool2_h_dim = 5
        pool2_w_dim = 5
        self.pool2 = nn.MaxPool2d(pool2_h_dim, pool2_w_dim)
        h_dim = h_dim/pool2_h_dim  # this layer is pooled as well
        w_dim = w_dim/pool2_w_dim 
        print("after pool2: ", h_dim, w_dim)
        h_dim = int(h_dim)
        w_dim = int(w_dim)
        

        # Standard linear layers
        self.lin_dims = h_dim*w_dim*16
        print("Lin dims: ", self.lin_dims)

        self.fc1 = nn.Linear(self.lin_dims, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)

        # Sigmoid out
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))    # same as (2,2)
        x = x.view(-1, self.lin_dims) 
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = self.sigmoid(x) ` # not needed if using BCEWithLogitsLoss
        return x


################################################################################
# Training
################################################################################
if __name__=="__main__":
    ## Run parameters
    file_prefix = "jsb_data"
    # file_path   = "/home/dahlbom/research/dmm_pitch/data_gen/"
    file_path   = "C:/Users/Beranek/Documents/dahlbom/dmm_pitch/cnn_front_end/"
    batch_size  = 8
    num_workers = 2
    num_epochs = 100 

    ## Set up network
    net = Net([600, 72])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, "\n")
    net.to(device)
    
    ## Set up loss fucntion and optimizer
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    ## Load data
    bach_dataset = auditoryDataset(file_path,
                                   file_prefix,
                                   transform=transforms.Compose([Renorm(),
                                                                 ToTensor()]))
    trainloader = DataLoader(bach_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers)

    ## Begin training
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data['image'], data['notes']
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:
                print('[%d, %5d] loss: %.8f' %
                        (epoch + 1, i+1, running_loss/2000))
                running_loss = 0.0

    print('Finished Training')

    # Now see what an output looks like...
    fig = plt.figure()
    for k in range(15):
        rand_idx = np.random.randint(0,len(bach_dataset))
        target = bach_dataset[rand_idx]['notes']
        in_image = bach_dataset[rand_idx]['image']
        output = net(in_image.unsqueeze(0).to('cuda'))
        sig = nn.Sigmoid()
        output = sig(output)
        output = output.to('cpu')
        ax = fig.add_subplot(3,5,k+1) 
        ax.plot(output[0].detach().numpy())
        ax.plot(target.numpy())
        ax.set_ylim([0,1.1])
    plt.show()
    
