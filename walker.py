#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 09:37:30 2020

@author: marcus
"""
import gym
import copy
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import TensorDataset, DataLoader
from collections import namedtuple
import random
import matplotlib.pyplot as plt

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'log_prob', 'done'))

class ReplayMemory():
    '''
    Memory to save state transistions for batch training.
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, sample):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = sample
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        #return random.sample(self.memory, batch_size)
        return Transition(*zip(*self.memory))
        
    def clear(self):
        self.memory = []
        self.position = 0
    
    def __len__(self):
        return len(self.memory)

    
class A2CNet(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, action_std):
        super(A2CNet,self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_space_dim = state_space_dim
        self.action_space_dim = action_space_dim
        
        action_layers = []
        action_layers += [nn.Linear(state_space_dim,64)]
        action_layers += [nn.Tanh()]
        action_layers += [nn.Linear(64,32)]
        action_layers += [nn.Tanh()]
        action_layers += [nn.Linear(32,self.action_space_dim)]
        action_layers += [nn.Tanh()]
        
        self.actor=nn.Sequential(*action_layers)
        
        critic_layers = []
        critic_layers += [nn.Linear(self.state_space_dim,64)]
        critic_layers += [nn.Tanh()]
        critic_layers += [nn.Linear(64,32)]
        critic_layers += [nn.Tanh()]
        critic_layers += [nn.Linear(32,1)]
        
        self.critic = nn.Sequential(*critic_layers)
        
        self.reduce_rate = None
        self.action_std = action_std
        self.action_var = torch.full((self.action_space_dim,), self.action_std*self.action_std).to(self.device)
        
        
    def act(self, state):
        action_mean = self.actor(state)
        cov_matrix = torch.diag(self.action_var).to(self.device)
        
        
        dist = MultivariateNormal(action_mean, cov_matrix)
        action = dist.sample().flatten()
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob
    
    
    def evaluate(self, old_state, old_action): 
        #TODO: Fix issue with wrong computation of log_probs
        action_mean = self.actor(old_state).squeeze()
        action_log_probs = torch.zeros(len(old_action), device=self.device)
        dist_entropy = torch.zeros(len(old_action), device=self.device)
        for i, am in enumerate(action_mean):
            action_var = self.action_var.expand_as(am)
            cov_mat = torch.diag(action_var).to(self.device)
            dist = MultivariateNormal(am, cov_mat)
            
            #probability of old action under new policy
           
            action_log_probs[i] = dist.log_prob(old_action[i])
            dist_entropy[i] = dist.entropy()
            
        state_value = self.critic(old_state)
        
        return torch.squeeze(state_value), action_log_probs, dist_entropy
    
    
    
    # def evaluate(self, old_state, old_action): 
    #     #TODO: Fix issue with wrong computation of log_probs
    #     action_mean = self.actor(old_state)
        
    #     action_var = self.action_var.expand_as(action_mean)
    #     cov_mat = torch.diag_embed(action_var).to(self.device)
    #     dist = MultivariateNormal(action_mean, cov_mat)
        
        
    #     #probability of old action under new policy
       
    #     action_log_probs = dist.log_prob(old_action)
    #     dist_entropy = dist.entropy()
    #     state_value = self.critic(old_state)
        
    #     return torch.squeeze(state_value), action_log_probs, dist_entropy
    
    def forward(self):
        raise NotImplementedError
        
    def reduce_std(self):
        if self.action_std >= 0.05:
            self.action_std -= self.reduce_rate
            self.action_var = torch.full((self.action_space_dim,), self.action_std*self.action_std).to(self.device)
    

class Trainer():
    def __init__(self, env, network, update_timestep = 2000, batch_size=512, gamma=0.99, epsilon=0.2, c1=0.5, c2=0.01, lr=0.01,
                 weight_decay=0.0, start_std=0.9, min_std=0.1):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.memory = ReplayMemory(update_timestep)
        self.env = env
        self.policy_net = network
        self.policy_net.to(self.device)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        
        self.update_timestep = update_timestep
        self.start_std = start_std
        self.min_std = min_std
        
    
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay)
        self.mse = nn.MSELoss()
        
        self.reward_log = []
        self.time_log = []
        self.num_updates = 0
        
    def ppo_update(self, num_epochs):
        
        self.num_updates += 1
        
        experience = self.memory.sample()
        exp_states = torch.stack(experience.state).float()
        exp_actions = torch.stack(experience.action)
        exp_rewards = experience.reward
        exp_dones = experience.done
        exp_log_probs = torch.stack(experience.log_prob)
        
        
        #calculate q-values
        q_values = []
        discounted_reward = 0
        for reward, done in zip(exp_rewards, exp_dones):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            q_values.insert(0, discounted_reward)
        
        
        q_values = torch.tensor(q_values, device=self.device)
        q_values = (q_values-q_values.mean())/(q_values.std()+1e-5)
        
        
        dataset = TensorDataset(exp_states, exp_actions, exp_log_probs, q_values)
        trainloader = DataLoader(dataset,batch_size=self.batch_size, shuffle=False)
        
        for _ in range(num_epochs):
            for state_batch, action_batch, log_probs_batch, q_value_batch in trainloader:
            
                #evaluate previous states
                state_values, log_probs, dist_entropy = self.policy_net.evaluate(state_batch, action_batch)
                
                
                # Calculate ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(log_probs - log_probs_batch.detach())
                    
                # Calculate Surrogate Loss:
                advantages = q_value_batch - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
                
                actor_loss = torch.min(surr1, surr2)
                critic_loss = self.mse(state_values, q_value_batch)
                
                # - because of gradient ascent
                loss = -actor_loss + self.c1*critic_loss - self.c2*dist_entropy
                
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
            
    def train(self, num_episodes, num_epochs, max_timesteps, render=False):
        #set rate to reduce the std of the actor
        self.policy_net.reduce_rate = (self.start_std-self.min_std)/num_episodes
    
        timestep = 0
        for i_episode in range(1, num_episodes+1):
            state = env.reset()
            running_reward = 0
            for i_timestep in range(max_timesteps):
                timestep += 1
                
                # compute action
                state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
                prev_state = state
                with torch.no_grad():
                    action, action_log_prob = self.policy_net.act(state)
                
                
                state, reward, done, _ = env.step(action.cpu().numpy())
                running_reward += reward
                transition = Transition(prev_state, action, reward, action_log_prob, done)
                self.memory.push(transition)
                
                #Update policy network
                if timestep % self.update_timestep == 0:
                    self.ppo_update(num_epochs)
                    print("Policy updated")
                    self.memory.clear()
                    # timestep=0
                    
                if render:
                    env.render()
         
                if done:
                    break
            
            print('Episode {} Done, \t length: {} \t reward: {}'.format(i_episode, i_timestep, running_reward))
            self.reward_log.append(running_reward)
            self.time_log.append(i_timestep)
            
            
            #reduce std to make actions less random over time
            self.policy_net.reduce_std()
            
    def plot_rewards(self):
        plt.plot(self.reward_log)
        #plt.plot(self.time_log, "episode length")
        #plt.legend()
        
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

num_episodes = 150
num_epochs = 40
max_timesteps = 1500
update_timestep = 4096
render = False
lr = 3e-4
weight_decay = 0.0


###############################################################################



env = gym.make('BipedalWalker-v3')
observation = env.reset()

#only for 1-dim and action state space
state_space_dim = env.observation_space.shape[0]
action_space_dim = env.action_space.shape[0]
network = A2CNet(state_space_dim, action_space_dim, action_std=0.5)

trainer = Trainer(env, network, update_timestep=update_timestep, lr=lr, weight_decay=weight_decay)
trainer.train(num_episodes=num_episodes, num_epochs=num_epochs, max_timesteps=max_timesteps, render=render)

test(env, network)



