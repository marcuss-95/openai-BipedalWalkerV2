#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:40:43 2020

@author: marcus
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory, Transition


class Trainer():
    def __init__(self, env, network, update_timestep = 2000, batch_size=512, gamma=0.99, epsilon=0.2, c1=0.5, c2=0.01, lr=0.01,
                 weight_decay=0.0,  min_std=0.1):
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
        self.min_std = min_std
        
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9,0.999))
        self.mse = nn.MSELoss()
        
        self.reward_log = []
        self.time_log = []
        self.num_updates = 0
        
    def ppo_update(self, num_epochs):
        
        self.num_updates += 1
        
        experience = self.memory.sample()
        #dimension of states need to be squeezed to not cause trouble in the  
        #log_probs computation in the evaluate function.
        exp_states = torch.stack(experience.state).squeeze().float()
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
        #self.policy_net.reduce_rate = (self.policy_net.action_std-self.min_std)/num_episodes
    
        timestep = 0
        for i_episode in range(1, num_episodes+1):
            state = self.env.reset()
            running_reward = 0
            for i_timestep in range(max_timesteps):
                timestep += 1
                
                # compute action
                state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
                prev_state = state
                with torch.no_grad():
                    action, action_log_prob = self.policy_net.act(state)
                
                
                state, reward, done, _ = self.env.step(action.cpu().numpy())
                running_reward += reward
                transition = Transition(prev_state, action, reward, action_log_prob, done)
                self.memory.push(transition)
                
                #Update policy network
                if timestep % self.update_timestep == 0:
                    self.ppo_update(num_epochs)
                    print("Policy updated")
                    self.memory.clear()
                    #reduce std of a network to make action less random
                    # self.policy_net.reduce_std()
                    
                    
                    timestep=0
                    
                if render:
                    env.render()
         
                if done:
                    break
            
            print('Episode {} Done, \t length: {} \t reward: {}'.format(i_episode, i_timestep, running_reward))
            self.reward_log.append(running_reward)
            self.time_log.append(i_timestep)
            
            
            
    def plot_rewards(self):
        plt.plot(self.reward_log)