# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:00:51 2024

@author: shinj
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import pandas as pd

import numpy as np





def Naviers_Stokes_Steady_Turbulent(x,y, mu, unet, vnet, pnet, epnet, knet):
    u = unet(x,y)
    v = vnet(x,y)
    p = pnet(x,y)
    k = knet(x,y)
    ep = epnet(x,y)
    mu_t = 0.09*k*k/ep
    # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    #u = output[:,0]
    #v = output[:,1]
    #p = output[:,2]

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
    v = vnet(x,y)
    p = pnet(x,y)
    
    # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    #u = output[:,0]
    #v = output[:,1]
    #p = output[:,2]

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
    # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    
    u = unet(x,y)
    v = vnet(x,y)
     
        
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.sum(), y, create_graph=True)[0]
    pde_cont = u_x + v_y
    return (pde_cont)


def k_Steady(x,y, unet, vnet, knet, epnet, mu):
    # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    
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
    # the dependent variable u is given by the network based on independent variables x,t
    ## Based on our f = du/dx - 2du/dt - u, we need du/dx and du/dt
    
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
