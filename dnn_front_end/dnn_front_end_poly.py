import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path
# project_directory = Path.cwd().parents[0]
# os.path.join(os.path.curdir
cur_dir = os.path.curdir
import pickle
import glob
import time


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
            print("Loading file {}".format(k+1))
            data = pickle.load(open(f, "rb"))
            y.append(data[0])
            x.append(data[1])
            if k+1 == 5: 
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
        # image /= 0.5*np.max(image)  #normalize between 0 and 1
        # image -= 1    # center at 0
        return {'image': image, 'notes': notes}

class ToTensor(object):

    def __call__(self, sample):
        image, notes = sample['image'], sample['notes']
        # image = image.transpose()
        # image = image.reshape((1,image.shape[0],image.shape[1]))
        return {'image': torch.from_numpy(image),
                'notes': torch.from_numpy(notes)}


################################################################################
# Network definition
################################################################################

class Net(nn.Module):

    @staticmethod
    def calc_dim_output(img_dim, filt_dim, stride, pad_dim):
        return (img_dim - filt_dim + 2*pad_dim)/stride + 1

    def __init__(self, ac_length=600, drop_p=0.0):
        super(Net, self).__init__()
        # Standard linear layers
        self.fc1 = nn.Linear(ac_length, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 88)

        # Sigmoid out
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(drop_p)

    def initialize(self):
        nn.init.xavier_uniform(self.linear.weight.data)
        self

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


################################################################################
# Training
################################################################################
if __name__=="__main__":
    ## Run parameters
    TRAIN = False
    TEST = True
    file_prefix_train = "poly_synth_data_train"
    file_prefix_test = "poly_synth_data_test"
    # file_path   = "C:/Users/Beranek/Documents/dahlbom/dmm_pitch/data_gen/"
    file_path = os.path.join(cur_dir + "/../data_gen/")
    # file_path   = str(project_directory / 'data_gen')
    save_prefix = "dnn_frontend_poly"
    # save_path   = "C:/Users/Beranek/Documents/dahlbom/dmm_pitch/dnn_front_end/saved_models/"
    # save_path   = str(project_directory / 'dnn_front_end' / 'saved_models/')
    save_path = os.path.join(cur_dir + "/../dnn_front_end/saved_models/")
    batch_size  = 10 
    num_workers = 0
    num_epochs = 200

    ## Set up network
    net = Net(ac_length=1024)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, "\n")
    net.to(device)
    
    ## Set up loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    ## Load data
    if TRAIN:
        print("Loading and transforming data...")
        bach_dataset = auditoryDataset(file_path,
                                       file_prefix_train,
                                       transform=transforms.Compose([ToTensor()]))
        print("Making data loader...")
        trainloader = DataLoader(bach_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)

        ## Begin training
        print("Finished. Starting training...")
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data['image'], data['notes']
                # print("Input shape: ", inputs.shape)
                # fig = plt.figure()
                # for k in range(inputs.shape[0]):
                #     ax = fig.add_subplot(2,int(inputs.shape[0]//2), k+1)
                #     ax.plot(inputs[k,:].numpy())
                # plt.show()
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            print('[%d] loss: %.8f' %
                    (epoch + 1, running_loss/2000), end="\t Time: ")
            print(time.strftime('%X'))
            running_loss = 0.0

        print('Finished Training')

        # Save the model
        torch.save(net.state_dict(), save_path + save_prefix + ".pt")

    # Now test the accuracy of the results
    if TEST:
        # Load model
        net.load_state_dict(torch.load(save_path + save_prefix + ".pt"))
        net.eval()

        # Loading the testing data
        print("Loading and transforming test data...")
        bach_dataset = auditoryDataset(file_path,
                                       file_prefix_test,
                                       transform=transforms.Compose([ToTensor()]))
        print("Making data loader for test data...")
        testloader = DataLoader(bach_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)
        with torch.no_grad():
            num_detected = 0
            num_in_gt = 0
            correctly_detected = 0
            for i, data in enumerate(testloader, 0):
                inputs, labels = data['image'], data['notes']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                note_est = torch.where(outputs > 0.5, 
                                       torch.ones_like(outputs), 
                                       torch.zeros_like(outputs))
                num_detected += torch.sum(note_est).item()
                correctly_detected += torch.sum(
                                        torch.where(note_est.double() == labels,
                                                    torch.ones_like(note_est),
                                                    torch.zeros_like(note_est)
                                                    ) * note_est).item()
                num_in_gt += torch.sum(labels).item()
            precision = correctly_detected/num_detected
            recall = correctly_detected/num_in_gt
            f_metric = 2.0/((1.0/precision) + (1.0/recall))
            print("Precision: {}".format(precision))
            print("Recall:    {}".format(recall))
            print("F-metric:  {}".format(f_metric))
                

    # Now see what an output looks like...
    fig = plt.figure()
    for k in range(21):
        rand_idx = np.random.randint(0,len(bach_dataset))
        target = bach_dataset[rand_idx]['notes']
        in_image = bach_dataset[rand_idx]['image']
        output = net(in_image.unsqueeze(0).to('cuda'))
        sig = nn.Sigmoid()
        output = sig(output)
        output = output.to('cpu')
        ax = fig.add_subplot(3,7,k+1) 
        ax.plot(output[0].detach().numpy())
        ax.plot(target.numpy())
        ax.set_ylim([0,1.1])
    plt.show()
    
