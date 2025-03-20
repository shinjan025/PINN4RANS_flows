# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 12:27:19 2024

@author: shinj
"""

from Model_archs import MLP_Net
from Trainer import Trainer
import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#define physics based nets vs variables
unet = MLP_Net() 
vnet = MLP_Net() 
pnet = MLP_Net() 
knet = MLP_Net() 
epnet = MLP_Net() 


unet = unet.to(device)
vnet = vnet.to(device)
pnet = pnet.to(device)
knet = knet.to(device)
epnet = epnet.to(device)

Trainer_1 = Trainer()

mu=0.001
max_iterations=200

Trainer_1.train_turbulent_random_collocation(mu, unet, vnet, pnet, epnet, knet, max_iterations)


modelA = MLP_Net()
modelB = MLP_Net()
optimizer1 = torch.optim.Adam(unet.parameters(),lr=0.001)
optimizer2 = torch.optim.Adam(vnet.parameters(),lr=0.001)

PATH = "model.pt"
checkpoint = torch.load(PATH, weights_only=True)
modelA.load_state_dict(checkpoint['modelu_state_dict'])
modelB.load_state_dict(checkpoint['modelv_state_dict'])
optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
