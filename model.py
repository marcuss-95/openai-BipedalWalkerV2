#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:36:42 2020

@author: marcus
"""
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class A2CNet(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, action_std):
        super(A2CNet,self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        
        body_layers = []
        body_layers += [nn.Linear(state_space_dim,64)]
        body_layers += [nn.Tanh()]
        body_layers += [nn.Linear(64,32)]
        body_layers += [nn.Tanh()]
        
        
        action_layers = []
        action_layers += [nn.Linear(32,32)]
        action_layers += [nn.Linear(32,self.action_space_dim)]
        #actions are constrained to be in the range [-1,1]
        action_layers += [nn.Tanh()]
        
        self.actor=nn.Sequential(*body_layers,*action_layers)
        
        critic_layers = []
        critic_layers += [nn.Linear(32,32)]
        critic_layers += [nn.Linear(32,1)]
        
        self.critic = nn.Sequential(*body_layers,*critic_layers)
    
        
        self.reduce_rate = None
        #self.action_std = nn.Parameter(torch.zeros(self.action_space_dim))
        #self.action_var = torch.full((self.action_space_dim,), self.action_std*self.action_std).to(self.device)
        
        self.action_var = nn.Parameter(torch.full((self.action_space_dim,), action_std*action_std), requires_grad=True).to(self.device)
        
    def act(self, state):
        action_mean = self.actor(state)
        cov_matrix = torch.diag(self.action_var).to(self.device)
        
        
        dist = MultivariateNormal(action_mean, cov_matrix)
        action = dist.sample().flatten()
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob
    
    def evaluate(self, old_state, old_action): 
        action_mean = self.actor(old_state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        #probability of old action under new policy
       
        action_log_probs = dist.log_prob(old_action)
        state_value = self.critic(old_state)
        dist_entropy = dist.entropy()
        
        return torch.squeeze(state_value), action_log_probs, dist_entropy
    
    def forward(self):
        raise NotImplementedError
        
    def reduce_std(self):
        if self.action_std >= 0.05:
            self.action_std -= self.reduce_rate
            self.action_var = torch.full((self.action_space_dim,), self.action_std*self.action_std).to(self.device)
    
