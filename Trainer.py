# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:35:05 2024

@author: shinj
"""
from Physics_models import Naviers_Stokes_Steady_Turbulent, Continuity_Steady, k_Steady, Epsilon_Steady, Naviers_Stokes_Steady
import torch
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd
import numpy as np


class Trainer():
    
    
 
    def __init__(
     self,
     #model: Pinn,
     #output_dir: Path = None,
     lr: float = 0.001,
     max_iterations: int = 1000,
     batch_size: int = 0.2,
    ):
            
      self.max_iterations=max_iterations
      self.batch_size=batch_size
      self.lr = lr
      
    
    
    
    def train_turbulent_random_collocation(self,mu, unet, vnet, pnet, epnet, knet, max_iterations):
        optimizer1 = torch.optim.Adam(unet.parameters(),lr=self.lr)
        optimizer2 = torch.optim.Adam(vnet.parameters(),lr=self.lr)
        optimizer3 = torch.optim.Adam(pnet.parameters(),lr=self.lr)
        optimizer4 = torch.optim.Adam(knet.parameters(),lr=self.lr)
        optimizer5 = torch.optim.Adam(epnet.parameters(),lr=self.lr)
        for epoch in range(max_iterations):
            optimizer1.zero_grad() # to make the gradients zero
            optimizer2.zero_grad() 
            optimizer3.zero_grad() 
            optimizer4.zero_grad() 
            optimizer5.zero_grad() 
            
            
            
            # Loss based on PDE
            x_collocation = np.random.uniform(low=0.0, high=2.0, size=(500,1))
            t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
            all_zeros = np.zeros((500,1))
            
            
            pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
            pt_y_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
            pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
            
            mse_cost_function = torch.nn.MSELoss()
            PDE_x, PDE_y = Naviers_Stokes_Steady_Turbulent(pt_x_collocation, pt_y_collocation, mu, unet, vnet, pnet, epnet, knet)# output of f(x,t)
            PDE_c = Continuity_Steady(pt_x_collocation, pt_y_collocation, unet, vnet)
            PDE_k = k_Steady(pt_x_collocation, pt_y_collocation, unet, vnet, knet, epnet, mu)
            PDE_ep = Epsilon_Steady(pt_x_collocation, pt_y_collocation, unet, vnet, knet, epnet, mu)
            mse_pde_1 = mse_cost_function(PDE_x, pt_all_zeros)
            mse_pde_2 = mse_cost_function(PDE_y, pt_all_zeros)
            mse_pde_3 = mse_cost_function(PDE_k, pt_all_zeros)
            mse_pde_4 = mse_cost_function(PDE_ep, pt_all_zeros)
            mse_pde_5 = mse_cost_function(PDE_c, pt_all_zeros)
           
            
            # Combining the loss functions
            loss = mse_pde_1 + mse_pde_2 +mse_pde_3 +mse_pde_4 +mse_pde_5
            
            
            loss.backward() # This is for computing gradients using backward propagation
            optimizer1.step() 
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            
            with torch.autograd.no_grad():
            	print(epoch,"Traning Loss:",loss.data)
            
        PATH = "model.pt"

        torch.save({
            'modelu_state_dict': unet.state_dict(),
            'modelv_state_dict': vnet.state_dict(),
            'optimizer1_state_dict': optimizer1.state_dict(),
            'optimizer2_state_dict': optimizer2.state_dict(),
            }, PATH)

    
    def train_turbulent_CFD_collocation(self,mu, Path, unet, vnet, pnet, epnet, knet, max_iterations):
        optimizer1 = torch.optim.Adam(unet.parameters(),lr=self.lr)
        optimizer2 = torch.optim.Adam(vnet.parameters(),lr=self.lr)
        optimizer3 = torch.optim.Adam(pnet.parameters(),lr=self.lr)
        optimizer4 = torch.optim.Adam(knet.parameters(),lr=self.lr)
        optimizer5 = torch.optim.Adam(epnet.parameters(),lr=self.lr)
        for epoch in range(max_iterations):
            optimizer1.zero_grad() # to make the gradients zero
            optimizer2.zero_grad() 
            optimizer3.zero_grad() 
            optimizer4.zero_grad() 
            optimizer5.zero_grad() 
            
            
            #data mini_batching
            
            df=pd.read_csv(Path)
            df_mini = df.sample(frac=0.2, Replace="False")
            x_collocation = df_mini["x"]
            y_collocation = df_mini["y"]
            all_zeros = np.zeros((len(x_collocation),1))
            
            
            
            pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
            pt_y_collocation = Variable(torch.from_numpy(y_collocation).float(), requires_grad=True).to(device)
            pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
            
            mse_cost_function = torch.nn.MSELoss()
            PDE_x, PDE_y = Naviers_Stokes_Steady_Turbulent(pt_x_collocation, pt_y_collocation, mu, unet, vnet, pnet, epnet, knet)# output of f(x,t)
            PDE_c = Continuity_Steady(pt_x_collocation, pt_y_collocation, unet, vnet)
            PDE_k = k_Steady(pt_x_collocation, pt_y_collocation, unet, vnet, knet, epnet, mu)
            PDE_ep = Epsilon_Steady(pt_x_collocation, pt_y_collocation, unet, vnet, knet, epnet, mu)
            mse_pde_1 = mse_cost_function(PDE_x, pt_all_zeros)
            mse_pde_2 = mse_cost_function(PDE_y, pt_all_zeros)
            mse_pde_3 = mse_cost_function(PDE_k, pt_all_zeros)
            mse_pde_4 = mse_cost_function(PDE_ep, pt_all_zeros)
            mse_pde_5 = mse_cost_function(PDE_c, pt_all_zeros)
           
            
            # Combining the loss functions
            loss = mse_pde_1 + mse_pde_2 +mse_pde_3 +mse_pde_4 +mse_pde_5
            
            
            loss.backward() # This is for computing gradients using backward propagation
            optimizer1.step() 
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            
            with torch.autograd.no_grad():
            	print(epoch,"Traning Loss:",loss.data)
                
        torch.save(unet.state_dict(), "C:/Users/shinj/Documents/PINNs")  
        torch.save(vnet.state_dict(), "C:/Users/shinj/Documents/PINNs")  
        torch.save(pnet.state_dict(), "C:/Users/shinj/Documents/PINNs")  
        torch.save(epnet.state_dict(), "C:/Users/shinj/Documents/PINNs")  
        torch.save(knet.state_dict(), "C:/Users/shinj/Documents/PINNs") 


    def train_laminar_random_collocation(self,mu, unet, vnet, pnet, max_iterations):
        optimizer1 = torch.optim.Adam(unet.parameters(),lr=self.lr)
        optimizer2 = torch.optim.Adam(vnet.parameters(),lr=self.lr)
        optimizer3 = torch.optim.Adam(pnet.parameters(),lr=self.lr)
        
        for epoch in range(max_iterations):
            optimizer1.zero_grad() # to make the gradients zero
            optimizer2.zero_grad() 
            optimizer3.zero_grad() 
            
                       
            # Loss based on PDE
            x_collocation = np.random.uniform(low=0.0, high=2.0, size=(500,1))
            t_collocation = np.random.uniform(low=0.0, high=1.0, size=(500,1))
            all_zeros = np.zeros((500,1))
            
            
            pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
            pt_y_collocation = Variable(torch.from_numpy(t_collocation).float(), requires_grad=True).to(device)
            pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
            
            mse_cost_function = torch.nn.MSELoss()
            PDE_x, PDE_y = Naviers_Stokes_Steady(pt_x_collocation, pt_y_collocation, mu, unet, vnet, pnet)# output of f(x,t)
            PDE_c = Continuity_Steady(pt_x_collocation, pt_y_collocation, unet, vnet)
            
            mse_pde_1 = mse_cost_function(PDE_x, pt_all_zeros)
            mse_pde_2 = mse_cost_function(PDE_y, pt_all_zeros)
            mse_pde_3 = mse_cost_function(PDE_c, pt_all_zeros)
           
            
            # Combining the loss functions
            loss = mse_pde_1 + mse_pde_2 +mse_pde_3 
            
            loss.backward() 
            optimizer1.step() 
            optimizer2.step()
            optimizer3.step()
           
            
            with torch.autograd.no_grad():
            	print(epoch,"Traning Loss:",loss.data)
                

        torch.save(unet.state_dict(), "C:/Users/shinj/Documents/PINNs")  
        torch.save(vnet.state_dict(), "C:/Users/shinj/Documents/PINNs")  
        torch.save(pnet.state_dict(), "C:/Users/shinj/Documents/PINNs")  
       

            

        
        