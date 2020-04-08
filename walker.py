#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 09:01:44 2020

@author: marcus
"""

from collections import deque


import gym
import copy
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


MAX_MEMORY_LENGTH = None


# class Memory():
#     def __init__(self):
#         self.states = deque([], maxlen = MAX_MEMORY_LENGTH)
#         self.actions = deque([], maxlen = MAX_MEMORY_LENGTH)
#         self.log_probs = deque([], maxlen = MAX_MEMORY_LENGTH)
#         self.rewards = deque([], maxlen = MAX_MEMORY_LENGTH)
#         self.dones = deque([], maxlen = MAX_MEMORY_LENGTH)
    
#     def clear(self):
#         self.states.clear()
#         self.action.clear()
#         self.log_probs.clear()
#         self.reward.clear()
#         self.dones.clear()
        
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.dones[:]



class A2CNet(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, action_std):
        super(A2CNet,self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        
        action_layers = []
        action_layers += [nn.Linear(state_space_dim,64)]
        action_layers += [nn.ReLU()]
        action_layers += [nn.Linear(64,32)]
        action_layers += [nn.ReLU()]
        action_layers += [nn.Linear(32,self.action_space_dim)]
        action_layers += [nn.Softmax(dim=-1)]
        
        self.actor=nn.Sequential(*action_layers)
        
        critic_layers = []
        critic_layers += [nn.Linear(self.state_space_dim,64)]
        critic_layers += [nn.ReLU()]
        critic_layers += [nn.Linear(64,32)]
        critic_layers += [nn.ReLU()]
        critic_layers += [nn.Linear(32,1)]
        
        self.critic = nn.Sequential(*critic_layers)
        
        self.action_var = torch.full((self.action_space_dim,), action_std*action_std).to(self.device)
        
        
    def act(self, state):
        action_mean = self.actor(state)
        cov_matrix = torch.diag(self.action_var).to(self.device)
        
        dist = MultivariateNormal(action_mean, cov_matrix)
        action = dist.sample().flatten()
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action, action_log_prob, dist_entropy
    
    def evaluate(self, old_state, old_action): 
        action_mean = self.actor(old_state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_log_probs = dist.log_prob(old_action)
        dist_entropy = dist.entropy()
        state_value = self.critic(old_state)
        
        return action_log_probs, torch.squeeze(state_value), dist_entropy
    
    def forward(self):
        raise NotImplementedError
    


class Trainer():
    def __init__(self, env, network, update_timestep = 4000, gamma=0.99, epsilon=0.2, c1=0.5, c2=0.01, lr=0.01):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = Memory()
        self.env = env
        self.policy_net = network
        self.policy_net.to(self.device)
        
        self.old_policy_net = copy.deepcopy(self.policy_net)
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        
        self.update_timestep = update_timestep
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        
    def ppo_update(self, num_epochs):
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.memory.rewards), reversed(self.memory.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        #Normalize
        rewards = torch.tensor(rewards).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-9)
        
        # convert list from memory to torch tensor
        old_states = torch.squeeze(torch.stack(self.memory.states).to(self.device), 1).detach()
        old_actions = torch.squeeze(torch.stack(self.memory.actions).to(self.device), 1).detach()
        old_log_probs = torch.squeeze(torch.stack(self.memory.log_probs), 1).to(self.device).detach()
        
        for _ in range(num_epochs):
            #evaluate previous actions
            log_probs, state_values, dist_entropy = self.policy_net.evaluate(old_states, old_actions)
            
            # Calculate ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(log_probs - old_log_probs.detach())
                
            # Calculate Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
            
            actor_loss = torch.min(surr1, surr2)
            critic_loss = self.mse(state_values, rewards)
            
            # - because of gradient ascent
            loss = -actor_loss + self.c1*critic_loss - self.c2*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        #Set the new policy to be the old one by copying the weights.
        self.old_policy_net.load_state_dict(self.policy_net.state_dict())
            
    def train(self, num_episodes, num_epochs, max_timesteps, render=False):
        
        timestep = 0
        for i_episode in range(1, num_episodes+1):
            state = env.reset()
            running_reward = 0
            for i_timestep in range(max_timesteps):
                timestep+=1
                
                state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
                
                action, action_log_prob, dist_entropy = self.old_policy_net.act(state)
                
                self.memory.states.append(state)
                self.memory.actions.append(action)
                self.memory.log_probs.append(action_log_prob)
                
                state, reward, done, _ = env.step(action.cpu().numpy())
                running_reward += reward
                
                self.memory.rewards.append(reward)
                self.memory.dones.append(done)
                
                #Update policy network
                if timestep % self.update_timestep == 0:
                    print("Policy updated")
                    self.ppo_update(num_epochs)
                    self.memory.clear()
                    
                if render:
                    env.render()
         
                if done:
                    print('Episode {} Done, \t length: {} \t reward: {}'.format(i_episode, i_timestep, running_reward))
                    break
                    
#%%              
def test(env, network):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = env.reset()
    
    sum_reward = 0
    i = 0
    while(i<500):
        env.render()
        action = network.actor(torch.tensor(state, dtype=torch.float, device=device))
        print(action)
        state, reward, done, _ = env.step(action.cpu().detach().numpy())
        sum_reward += reward
        i+=1
    print("Testrun. Reward: {}".format(sum_reward))
    env.close()
#%%          


env = gym.make('BipedalWalker-v3')
observation = env.reset()

#only for 1-dim and action state space
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.shape[0]
network = A2CNet(state_space_dim, action_space_dim, action_std=0.5)

trainer = Trainer(env, network, lr=1e-5)
trainer.train(num_episodes=100, num_epochs=80, max_timesteps=1500)

test(env, network)



