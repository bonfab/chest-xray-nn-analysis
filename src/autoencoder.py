#!/usr/bin/env python3
#
#
# Source: https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential( # like the Composition layer you built
            nn.Conv2d(1, 16, 5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



def train(model, data, num_epochs=5, batch_size=64, learning_rate=1e-3):
    torch.manual_seed(42)
    # Loss function and optimizer
    criterion = nn.MSELoss() # mean square error loss
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=1e-5) # <--
    train_loader = torch.utils.data.DataLoader(data,
                                               batch_size=batch_size,
                                               shuffle=True)

    # Train the network
    outputs = []
    for epoch in range(num_epochs):
        for data in train_loader:
            img, _ = data
            recon = model(img)
            loss = criterion(recon, img)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch:{}, Loss:{:.4f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, img, recon),)
    return outputs

def export_embeeding_to_csv(model, data, dir_name, batch_size=64):
    data_loader = torch.utils.data.DataLoader(data, 
                                              batch_size=batch_size,
                                              shuffle=False)

    model.eval()
    arr = np.empty((9000,64))
    label_arr = np.empty((9000,5))
    
    for i, (d, l) in enumerate(data_loader):
        arr[i*64:(i+1)*64] = model.encoder(d).flatten(start_dim=2).view(64,64).detach().numpy()
        label_arr[i*64:(i+1)*64] = l.detach().numpy()
    
    df = pd.DataFrame(arr, columns=list(np.arange(64))+['l1','l2','l3','l4','l5'])
    df.iloc[:,:64] = arr
    df.iloc[:,64:] = label_arr
    
    print(df)

def visualize_training(outputs, max_epochs, dir_name):
    for k in range(0, max_epochs, 5):
        plt.figure(figsize=(9, 2))
        plt.gray()
        imgs = outputs[k][1].detach().numpy()
        recon = outputs[k][2].detach().numpy()
        for i, item in enumerate(imgs):
            if i >= 9: break
            plt.subplot(2, 9, i+1)
            plt.imshow(item[0])
            
            #os.makedirs(f'../results/autoencoder/{dir_name}', exist_ok=True)
            #plt.savefig(f'../results/autoencoder/{dir_name}/img-{k}-{i}.png')

        for i, item in enumerate(recon):
            if i >= 9: break
            plt.subplot(2, 9, 9+i+1)
            plt.imshow(item[0])
            
            os.makedirs(f'../results/autoencoder/{dir_name}', exist_ok=True)
            #plt.savefig(f'../results/autoencoder/{dir_name}/recons-{k}-{i}.png')
