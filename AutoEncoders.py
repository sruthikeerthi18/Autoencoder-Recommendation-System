# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python',encoding='latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python',encoding='latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python',encoding='latin-1')

train=pd.read_csv('ml-100k/u1.base',delimiter='\t')
train=np.array(train,dtype=int)
test=pd.read_csv('ml-100k/u1.test',delimiter='\t')
test=np.array(test,dtype=int)

nb_users=int(max(max(train[:,0]),max(test[:,0])))
nb_movies=int(max(max(train[:,1]),max(test[:,1])))


def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]
        id_ratings = data[:,2][data[:,0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

train=convert(train)
test=convert(test)

train=torch.FloatTensor(train)
test=torch.FloatTensor(test)        
#******* END OF PREPROCESSING ********

# AutoEncoder
class SAE(nn.Module):
    def __init__(self):
        super(SAE,self).__init__()
        self.fc1=nn.Linear(nb_movies,20)
        self.fc2=nn.Linear(20,10)
        self.fc3=nn.Linear(10,20)
        self.fc4=nn.Linear(20,nb_movies)
        self.activation=nn.Sigmoid()
    def forward(self,x):
        x=self.activation(self.fc1)
        x=self.activation(self.fc2)
        x=self.activation(self.fc3)
        x=self.fc4()
        return x
sae=SAE()
criterion=nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        x = Variable(train[id_user]).unsqueeze(0)
        x=torch.Tensor(x)
        target = x.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(x)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data[0]*mean_corrector)
            s += 1.
            optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(train[id_user]).unsqueeze(0)
    target = Variable(test[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0]*mean_corrector)
        s += 1.
print('test loss: '+str(test_loss/s))    