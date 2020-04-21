#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:37:41 2020

@author: marcus
"""
from collections import namedtuple

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