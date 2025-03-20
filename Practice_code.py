# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:07:20 2024

@author: shinj
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd

import numpy as np

# We consider Net as our solution u_theta(x,t)


class uNet(nn.Module):
    def __init__(self):
        super(uNet, self).__init__()
        self.hidden_layer1 = nn.Linear(2,5)
        self.hidden_layer2 = nn.Linear(5,5)
        self.hidden_layer3 = nn.Linear(5,5)
        self.hidden_layer4 = nn.Linear(5,5)
        self.hidden_layer5 = nn.Linear(5,5)
        self.output_layer = nn.Linear(5,1)

    def forward(self, x,t):
        inputs = torch.cat([x,t],axis=1) # combined two arrays of 1 columns each to one array of 2 columns
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out) ## For regression, no activation is used in output layer
        return output



### (2) Model
unet = uNet()
unet = unet.to(device)

vnet = uNet()
vnet = vnet.to(device)

pnet = uNet()
pnet = pnet.to(device)

knet = uNet()
knet = knet.to(device)

epnet = uNet()
epnet = epnet.to(device)

mse_cost_function = torch.nn.MSELoss() # Mean squared error
optimizer1 = torch.optim.Adam(unet.parameters())
optimizer2 = torch.optim.Adam(vnet.parameters())
optimizer3 = torch.optim.Adam(pnet.parameters())
optimizer4 = torch.optim.Adam(knet.parameters())
optimizer5 = torch.optim.Adam(epnet.parameters())


## PDE loss functions 
def Naviers_Stokes_Steady_Turbulent(x,y, mu, unet, vnet, pnet, epnet, knet):
    u = unet(x,y)
    v = unet(x,y)
    p = unet(x,y)
    k = knet(x,y)
    ep = epnet(x,y)
    mu_t = 0.09*k*k/ep
  

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_x_x = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_y_y = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0] 
    
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_x_x = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    v_y_y = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0] 
    
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    
    pdex = u*u_x+u*v_x - p_x - (mu+mu_t)*(u_x_x) -(mu+mu_t)*(u_y_y)
    pdey = v*u_y+v*v_y - p_y - (mu+mu_t)*(v_x_x) - (mu+mu_t)*(v_y_y)
    return (pdex, pdey)


def Naviers_Stokes_Steady(x,y, mu, unet, vnet, pnet):
    u = unet(x,y)
    v = unet(x,y)
    p = unet(x,y)
    
 

    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_x_x = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    u_y = torch.autograd.grad(u.sum(), y, create_graph=True)[0]
    u_y_y = torch.autograd.grad(u_y.sum(), y, create_graph=True)[0] 
    
    v_x = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
    v_x_x = torch.autograd.grad(v_x.sum(), x, create_graph=True)[0]
    
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    v_y_y = torch.autograd.grad(v_y.sum(), y, create_graph=True)[0] 
    
    p_x = torch.autograd.grad(p.sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.sum(), y, create_graph=True)[0]
    
    pdex = u*u_x+u*v_x - p_x - (mu)*(u_x_x) -(mu)*(u_y_y)
    pdey = v*u_y+v*v_y - p_y - (mu)*(v_x_x) - (mu)*(v_y_y)
    return (pdex, pdey)


def Continuity_Steady(x,y, unet, vnet):
    
    
    u = unet(x,y)
    v = vnet(x,y)
     
        
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    pde_cont = u_x + v_y
    return (pde_cont)


def k_Steady(x,y, unet, vnet, knet, epnet, mu):
    
    u = unet(x,y)
    v = vnet(x,y)
    k = knet(x,y)
    ep = epnet(x,y)
    #p = output[:,2]
   
    mu_t = 0.09*k*k/ep
    
    u_x = torch.autograd.grad(k.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(k.sum(), y, create_graph=True)[0]
    v_y = torch.autograd.grad(ep.sum(), y, create_graph=True)[0]
    v_x = torch.autograd.grad(ep.sum(), x, create_graph=True)[0]
    k_x = torch.autograd.grad(k.sum(), x, create_graph=True)[0]
    k__x = torch.autograd.grad(k_x.sum(), x, create_graph=True)[0]
    k_y = torch.autograd.grad(k.sum(), y, create_graph=True)[0]
    k__y = torch.autograd.grad(k_y.sum(), y, create_graph=True)[0]
     
    P_k = mu_t*(  2*(u_x)**2+ 2*(v_y)**2 + (u_y)**2+(v_x)**2+(2*u_y*v_x))
        
    sig_k=1
    sig=1    
          
    pde_k= k_x + v*k_y - (mu + mu_t/sig_k)*k__x- (mu + mu_t/sig_k)*k__y -P_k + ep
    return (pde_k)

def Epsilon_Steady(x,y, unet, vnet, knet, epnet, mu):
    
    
    u = unet(x,y)
    v = vnet(x,y)
    k = knet(x,y)
    ep = epnet(x,y)
    #p = output[:,2]
    mu_t = 0.09*k*k/ep
    
    u_x = torch.autograd.grad(k.sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(k.sum(), y, create_graph=True)[0]
    v_y = torch.autograd.grad(ep.sum(), y, create_graph=True)[0]
    v_x = torch.autograd.grad(ep.sum(), x, create_graph=True)[0]
    ep_x = torch.autograd.grad(ep.sum(), x, create_graph=True)[0]
    ep__x = torch.autograd.grad(ep_x.sum(), x, create_graph=True)[0]
    ep_y = torch.autograd.grad(ep.sum(), y, create_graph=True)[0]
    ep__y = torch.autograd.grad(ep_y.sum(), y, create_graph=True)[0]
    P_k = mu_t*(  2*(u_x)**2+ 2*(v_y)**2 + (u_y)**2+(v_x)**2+(2*u_y*v_x))
    C_ep1=1
    C_ep2=1
    sig_k=1
    pde_eps = ep_x + v*ep_y - (mu + mu_t/sig_k)*ep__x- (mu + mu_t/sig_k)*ep__y - (C_ep1 * P_k - C_ep2 * ep) * ep / (k + 1e-3)
    return (pde_eps)

def Omega_Steady(x,y, unet, vnet):
    # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    
    u = unet(x,y)
    v = vnet(x,y)
    #p = output[:,2]
    
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    pde_cont = u_x + v_y
    return (pde_cont)


## Data from Boundary Conditions
# u(x,0)=6e^(-3x)
## BC just gives us datapoints for training

# BC tells us that for any x in range[0,2] and time=0, the value of u is given by 6e^(-3x)
# Take say 500 random numbers of x
x_bc = np.random.uniform(low=0.0, high=2.0, size=(500,1))
t_bc = np.zeros((500,1))
# compute u based on BC
u_bc = 6*np.exp(-3*x_bc)



### (3) Training / Fitting
iterations = 20000
previous_validation_loss = 99999999.0
mini_batch_size = 0.2
mu=0.001
for epoch in range(iterations):
    optimizer1.zero_grad() # to make the gradients zero
    optimizer2.zero_grad() 
    optimizer3.zero_grad() 
    optimizer4.zero_grad() 
    optimizer5.zero_grad() 
    
    # Loss based on boundary conditions
    pt_x_bc = Variable(torch.from_numpy(x_bc).float(), requires_grad=False).to(device)
    pt_t_bc = Variable(torch.from_numpy(t_bc).float(), requires_grad=False).to(device)
    pt_u_bc = Variable(torch.from_numpy(u_bc).float(), requires_grad=False).to(device)
    
    #net_bc_out = net(pt_x_bc, pt_t_bc) # output of u(x,t)
    #mse_u = mse_cost_function(net_bc_out, pt_u_bc)
    
    #data mini_batching
    
    #df=pd.read_csv("data_file.csv")
    #df_mini = df.sample(frac=0.2, Replace="False")
    #x_collocation = df_mini["x"]
    #y_collocation = df_mini["y"]
    
    # Loss based on PDE
    x_collocation = np.random.uniform(low=0.0, high=2.0, size=(500,1))
    t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
    all_zeros = np.zeros((500,1))
    
    
    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_y_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    
    PDE_x, PDE_y = Naviers_Stokes_Steady_Turbulent(pt_x_collocation, pt_y_collocation, mu, unet, vnet, pnet, epnet, knet)# output of f(x,t)
    PDE_c = Continuity_Steady(pt_x_collocation, pt_y_collocation, unet, vnet)
    PDE_k = k_Steady(pt_x_collocation, pt_y_collocation, unet, vnet, knet, epnet, mu)
    PDE_ep = Epsilon_Steady(pt_x_collocation, pt_y_collocation, unet, vnet, knet, epnet, mu)
    mse_pde_1 = mse_cost_function(PDE_x, pt_all_zeros)
    mse_pde_2 = mse_cost_function(PDE_y, pt_all_zeros)
    mse_pde_3 = mse_cost_function(PDE_k, pt_all_zeros)
    mse_pde_4 = mse_cost_function(PDE_ep, pt_all_zeros)
   
    
    # Combining the loss functions
    loss = mse_pde_1 + mse_pde_2 +mse_pde_3 +mse_pde_4
    
    
    loss.backward() # This is for computing gradients using backward propagation
    optimizer1.step() # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
    optimizer2.step()
    optimizer3.step()
    optimizer4.step()
    optimizer5.step()
    
    with torch.autograd.no_grad():
    	print(epoch,"Traning Loss:",loss.data)
        





    


    
