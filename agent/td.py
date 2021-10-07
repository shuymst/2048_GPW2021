import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from utils import *

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class TDAgent:
    def __init__(self, gamma, lr):
        self.model = ValueNet().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.MseLoss = nn.MSELoss()
        self.buffer_next_rewards = []
        self.buffer_afterstates = []
        self.buffer_next_afterstates = []
        self.gamma = gamma
    
    def select_action(self, state):
        action = self.model.act(state)
        return action
    
    def update(self):
        batch_afterstate_values, batch_next_afterstate_values = self.model.evaluate(self.buffer_afterstates, self.buffer_next_afterstates)
        batch_rewards = torch.tensor(self.buffer_next_rewards[1:]).to(device)
        value_targets = batch_rewards + self.gamma * batch_next_afterstate_values.detach()
        loss = 0.5 * self.MseLoss(value_targets, batch_afterstate_values)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.clear()

    def clear(self):
        del self.buffer_afterstates[:]
        del self.buffer_next_afterstates[:]
        del self.buffer_next_rewards[:]