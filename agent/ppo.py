import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
from utils import *

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class PPO:
    def __init__(self, layer_num, channel_num, ppo_type, use_bn, lr, gamma, lambd, K_epochs, eps_clip):
        self.gamma = gamma
        self.lambd = lambd
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.buffer = RolloutBuffer()
        self.ppo_type = ppo_type
        if ppo_type == "normal":
            self.model = NormalActorCritic(layer_num, channel_num, use_bn).to(device)
        else:
            self.model = AfterstateActorCritic(layer_num, channel_num, use_bn).to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = lr, momentum=0.9, weight_decay=1e-4)
        if ppo_type == "normal":
            self.model_old = NormalActorCritic(layer_num, channel_num, use_bn).to(device)
        else:
            self.model_old = AfterstateActorCritic(layer_num, channel_num, use_bn).to(device)
        self.model_old.load_state_dict(self.model.state_dict())
        self.model_old.eval()
        self.model.train()
        self.MseLoss = nn.MSELoss()
    
    def select_action(self, state):
        with torch.no_grad():
            if self.ppo_type == "normal":
                action, action_logprob = self.model_old.act(state)
            else:
                action, action_logprob = self.model_old.act(state, self.buffer.afterstates, self.buffer.afterstate_rewards)
            state = to_3d(state)
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            return action
    
    def update(self):
        discounted_advantage = 0
        rewards = torch.tensor(self.buffer.rewards, dtype = torch.float).detach().to(device)
        old_states = torch.stack(self.buffer.states, dim=0).detach().to(device)
        old_afterstates = torch.stack(self.buffer.afterstates, dim=0).detach().to(device)
        old_afterstate_rewards = torch.tensor(self.buffer.afterstate_rewards).detach().to(device)
        old_actions = torch.tensor(self.buffer.actions).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs), dim=0).detach().to(device)
        for _ in range(self.K_epochs):
            if self.ppo_type == "normal":
                logprobs, state_values = self.model.evaluate(old_states, old_actions)
            else:
                logprobs, state_values = self.model.evaluate(old_states, old_actions, old_afterstates, old_afterstate_rewards)
            state_values = torch.squeeze(state_values)
            next_state_values = state_values.clone().detach()
            next_state_values = next_state_values[1:]
            next_state_values = torch.cat([next_state_values, torch.tensor([0.]).to(device)])
            for i in range(len(self.buffer.is_terminals)):
                if self.buffer.is_terminals[i]:
                    next_state_values[i] = 0.
            td_errors = rewards + self.gamma * next_state_values - state_values.detach()
            value_targets = rewards + self.gamma * next_state_values
            advantages = []
            discounted_advantage = 0
            for td_error, is_terminal in zip(reversed(td_errors), reversed(self.buffer.is_terminals)):
                if is_terminal:
                    discounted_advantage = 0
                discounted_advantage = td_error + self.gamma * self.lambd * discounted_advantage
                advantages.insert(0, discounted_advantage)
            advantages = torch.tensor(advantages, dtype = torch.float).detach().to(device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, value_targets)
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        self.model_old.load_state_dict(self.model.state_dict())
        self.buffer.clear()