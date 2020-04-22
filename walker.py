#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:37:30 2020

@author: marcus
"""
import gym
import copy
import torch

from model import A2CNet
from trainer import Trainer
    


        
#%%              
def test(env, network):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = env.reset()
    
    sum_reward = 0
    i = 0
    while(i<500):
        env.render()
        action, _ = network.act(torch.tensor(state, dtype=torch.float, device=device))
        # print(action)
        state, reward, done, _ = env.step(action.cpu().detach().numpy())
        if not done:
            sum_reward += reward
        i+=1
    print("Testrun. Reward: {}".format(sum_reward))
    env.close()
#%%          

################################HYPERPARAMETERS################################

batch_size = 512
num_episodes = 100
num_epochs = 70
max_timesteps = 1500
update_timestep = 4000
render = False
lr = 1e-4
weight_decay = 0.0
c2 = 0.01
std_trainable = False
###############################################################################



env = gym.make('BipedalWalker-v3')
observation = env.reset()

#only for 1-dim and action state space
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.shape[0]
network = A2CNet(state_space_dim, action_space_dim, action_std=0.5, std_trainable=std_trainable)

trainer = Trainer(env, network, update_timestep=update_timestep, lr=lr, weight_decay=weight_decay, batch_size=batch_size, c2=c2)
trainer.train(num_episodes=num_episodes, num_epochs=num_epochs, max_timesteps=max_timesteps, render=render)

test(env, network)



