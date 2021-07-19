import torch
import torch.nn as nn
from torch.nn import functional
import torchvision
import numpy as np
from tqdm.notebook import tqdm
import models
from sklearn.metrics import roc_auc_score
import numpy as np
from time import time
from datetime import datetime
import pandas as pd


def total_validation_loss_2class(model, val_loader, criterion):

    with torch.set_grad_enabled(False):
        val_loss = 0
        count = 0
        for val_x, val_y in val_loader:
            x, y = val_x.to(device), val_y.to(device)
            count += y.shape[0]
            val_loss += criterion(model(x), y)

        empirical_loss = val_loss.item() / count
        print("Empirical Validation Loss: {}".format(empirical_loss))

    return empirical_loss


def simple_train_2class(
    train_loader,
    val_loader,
    num_pathologies,
    input_size,
    model=models.testNet_2class,
    EPOCHS=10,
):

    """
    A simple wrapper for training. Useful for doing functionality testing and sanity checks.
    
    Args:
        train_loader: The dataloader that loads the training data
        
        val_loader: Dataloader that loads the validation data
        
        num_pathologies: Number of pathologies in the data set. Model adjusts accordingly.
        
        input_size: Dimension of Tensors loaded by dataloaders. Example: (256, 3)
    
    """
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")

    model = model(num_pathologies, input_size).to(device)
    print(model)
    print()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    model.eval()
    total_validation_loss_2class(model, val_loader, criterion)
    

    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0
        count = 0
        
        with tqdm(train_loader) as pbar:
            for i, batch in enumerate(pbar):

                x, y = batch

                x = x.to(device)
                y = y.to(device)
            
                batch_loss = criterion(model(x), y)
                train_loss += batch_loss.item()
                count += x.shape[0]
                
                pbar.set_postfix(loss=batch_loss.item()/y.shape[0])
            
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        print(f'Epoch {epoch+1}')
        print('Train Loss: {}'.format(train_loss/count))
        model.eval()
        total_validation_loss_2class(model, val_loader, criterion)

                
def total_validation_loss_ROCAUC_2class(model, val_loader, val_size, num_pathologies, criterion):
    
    predictions = np.empty((val_size, num_pathologies+1))
    labels = np.empty((val_size, num_pathologies+1))
    sigmoid = nn.Sigmoid()
    
    with torch.set_grad_enabled(False):
        val_loss = 0
        count = 0
        count_prev = 0
        for val_x, val_y in val_loader:
            x, y = val_x.to(device), val_y.to(device)
            count += y.shape[0]
            y_hat = model(x)
            val_loss += criterion(y_hat, y)
            predictions[count_prev:count] = sigmoid(y_hat).cpu().numpy()
            labels[count_prev:count] = val_y.numpy()
            
            count_prev = count
                
        empirical_loss = val_loss.item()/count
        print('Validation Loss: {}'.format(empirical_loss))
        auc = roc_auc_score(labels, predictions, average='weighted')
        print(f'Validation AUC: {auc}')
    return empirical_loss, auc
                
def train_2class(train_loader, val_loader, val_size, num_pathologies, input_size, model=models.testNet_2class, EPOCHS=10):
    
    torch.manual_seed(0)
    
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
        
    
    model = model(num_pathologies, input_size).to(device)
    print(model)
    print()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    model.eval()
    total_validation_loss_ROCAUC_2class(model, val_loader, val_size, num_pathologies, criterion)
    
    stats = np.empty((3,EPOCHS))
    
    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0
        count = 0
        
        with tqdm(train_loader) as pbar:
            for i, batch in enumerate(pbar):
            
                x, y = batch
            
                x = x.to(device)
                y = y.to(device)
            
                batch_loss = criterion(model(x), y)
                train_loss += batch_loss.item()
                count += x.shape[0]
                
                pbar.set_postfix(loss=batch_loss.item()/y.shape[0])
            
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        print(f'Epoch {epoch+1}')
        print('Train Loss: {}'.format(train_loss/count))
        model.eval()
        val_loss, auc = total_validation_loss_ROCAUC_2class(model, val_loader, val_size, num_pathologies, criterion)
        stats[:,epoch] = np.array([train_loss/count, val_loss, auc])
        
    return stats, model


def train_save_2class(train_loaders, val_loaders, names, val_size, num_pathologies, input_size, model, EPOCHS=20, save_name=""):
    
    stats = pd.DataFrame(columns=list(range(EPOCHS))+['Name', 'Metric'], index=np.arange(3*len(names)))
    
    for i in range(len(names)):
        
        stats.iloc[i*3:(i+1)*3].loc[:,list(range(EPOCHS))], model = train_2class(train_loaders[i], val_loaders[i], val_size, num_pathologies, input_size, model, EPOCHS)
        stats.loc[i*3:(i+1)*3-1,'Name'] = names[i]
        stats.loc[i*3:(i+1)*3-1,'Metric'] = ['Train-loss', 'Val-loss', 'AUC']
        
        t = datetime.utcfromtimestamp(time()).strftime("%Y-%m-%d_%H-%M-%S")
        
        print(model)
        
        if save_name == "":
            torch.save(model.state_dict(), '../trained/reduced-'+ names[i] + '-' + t + '.pt')
        else:
            torch.save(model.state_dict(), '../trained/reduced-' + names[i] + '-' + save_name + '.pt')

    t = datetime.utcfromtimestamp(time()).strftime("%Y-%m-%d_%H-%M-%S")
    stats.to_csv('../stats/' + t + '.csv', index=False)


def criterion(y_hat, y):
    cross_entropy = nn.BCEWithLogitsLoss()
    
    #print(y_hat.shape)
    #print(y_hat[0].shape)
    #print(y_hat[:,0].shape)
    
    loss = cross_entropy(y_hat[:,0], y[:,0])
    
    for i in range(1, y_hat.shape[1]):
        loss += cross_entropy(y_hat[:,i], y[:,i])
    
    return loss, y_hat[0,0], y[0,0]


def prediction(y_hat):
    
    sigmoid = nn.Sigmoid()
    
    y_hat_c = y_hat.cpu()
    del y_hat
    
    out = np.empty(y_hat_c.shape)
    
    for i in range(y_hat_c.shape[1]):
        out[:,i] = sigmoid(y_hat_c[:,i])
    
    return out
    
    
def total_validation_loss_ROCAUC_2class_mloss(model, val_loader, val_size, num_pathologies):
    
    predictions = np.empty((val_size, num_pathologies+1))
    labels = np.empty((val_size, num_pathologies+1))
    
    with torch.set_grad_enabled(False):
        val_loss = 0
        count = 0
        count_prev = 0
        for val_x, val_y in val_loader:
            x, y = val_x.to(device), val_y.to(device)
            count += y.shape[0]
            y_hat = model(x)
            l, p, t = criterion(y_hat, y)
            val_loss += l
            predictions[count_prev:count] = prediction(y_hat)
            labels[count_prev:count] = val_y.numpy()
            
            count_prev = count
                
        empirical_loss = val_loss.item()/count
        print('Validation Loss: {}'.format(empirical_loss))
        auc = roc_auc_score(labels, predictions, average='weighted')
        print(f'Validation AUC: {auc}')
    return empirical_loss, auc




def train_2class_mloss(train_loader, val_loader, val_size, num_pathologies, input_size, model=models.testNet_2class, EPOCHS=10):
    
    torch.manual_seed(0)
    
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        device = torch.device("cpu")
        print("Running on the CPU")
        
    
    model = model(num_pathologies, input_size).to(device)
    print(model)
    print()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.eval()
    #total_validation_loss_ROCAUC_2class_mloss(model, val_loader, val_size, num_pathologies)
    
    stats = np.empty((3,EPOCHS))
    
    sigmoid = nn.Sigmoid()
    
    for epoch in range(EPOCHS):
        
        model.train()
        train_loss = 0
        count = 0
        
        with tqdm(train_loader) as pbar:
            for i, batch in enumerate(pbar):
            
                x, y = batch
            
                x = x.to(device)
                y = y.to(device)
            
                batch_loss, p, t = criterion(model(x), y)
                train_loss += batch_loss.item()
                count += x.shape[0]
                
                pbar.set_postfix(loss=batch_loss.item()/y.shape[0], pred=abs(sigmoid(p).item()-t.item()))
            
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        
        torch.cuda.empty_cache()
        
        print(f'Epoch {epoch+1}')
        print('Train Loss: {}'.format(train_loss/count))
        model.eval()
        val_loss, auc = total_validation_loss_ROCAUC_2class_mloss(model, val_loader, val_size, num_pathologies)
        stats[:,epoch] = np.array([train_loss/count, val_loss, auc])
        torch.cuda.empty_cache()
        
    return stats, model

def train_save_2class_mloss(train_loaders, val_loaders, names, val_size, num_pathologies, input_size, model, EPOCHS=20, save_name=""):
    
    stats = pd.DataFrame(columns=list(range(EPOCHS))+['Name', 'Metric'], index=np.arange(3*len(names)))
    
    for i in range(len(names)):
        
        stats.iloc[i*3:(i+1)*3].loc[:,list(range(EPOCHS))], model = train_2class_mloss(train_loaders[i], val_loaders[i], val_size, num_pathologies, input_size, model, EPOCHS)
        stats.loc[i*3:(i+1)*3-1,'Name'] = names[i]
        stats.loc[i*3:(i+1)*3-1,'Metric'] = ['Train-loss', 'Val-loss', 'AUC']
        
        t = datetime.utcfromtimestamp(time()).strftime("%Y-%m-%d_%H-%M-%S")
        
        print(model)
        
        if save_name == "":
            torch.save(model.state_dict(), '../trained/reduced-'+ names[i] + '-' + t + '.pt')
        else:
            torch.save(model.state_dict(), '../trained/reduced-' + names[i] + '-' + t + save_name + '.pt')

    t = datetime.utcfromtimestamp(time()).strftime("%Y-%m-%d_%H-%M-%S")
    stats.to_csv('../stats/' + t + '.csv', index=False)
    
        
        
