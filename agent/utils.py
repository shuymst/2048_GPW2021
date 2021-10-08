import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
import torch.nn as nn
import random
import numpy as np
from collections import namedtuple, deque
import sys
sys.path.append("..")
import gym_2048

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def right(state):
    table_tmp = state.copy()
    double_list = []
    reward = 0
    double_table = torch.zeros(size=(4,4))
    flag = False
    for i in range(4):
        for j in range(2, -1, -1):
            if table_tmp[i][j] == 0:
                continue
            for k in range(j+1, 4):
                if table_tmp[i][k] == 0 and k == 3:
                    table_tmp[i][k] = table_tmp[i][j]
                    table_tmp[i][j] = 0
                    flag = True
                elif table_tmp[i][k] == 0 and k != 3:
                    continue
                elif table_tmp[i][k] == table_tmp[i][j]:
                    double_list.append((i,k,table_tmp[i][k]*2))
                    table_tmp[i][k] = -1
                    table_tmp[i][j] = 0
                    flag = True
                    break
                else:
                    if j == k-1:
                        break
                    else:
                        table_tmp[i][k-1] = table_tmp[i][j]
                        table_tmp[i][j] = 0
                        flag = True
                        break
    for double in double_list:
        table_tmp[double[0]][double[1]] = double[2]
        reward += double[2]
        double_table[double[0]][double[1]] = 1
    return table_tmp, reward, flag, double_table

def left(state):
    table_tmp = state.copy()
    double_list = []
    reward = 0
    flag = False
    double_table = torch.zeros(size=(4,4))
    for i in range(4):
        for j in range(1, 4):
            if table_tmp[i][j] == 0:
                continue
            for k in range(j-1, -1, -1):
                if table_tmp[i][k] == 0 and k == 0:
                    table_tmp[i][k] = table_tmp[i][j]
                    table_tmp[i][j] = 0
                    flag = True
                elif table_tmp[i][k] == 0 and k != 0:
                    continue
                elif table_tmp[i][k] == table_tmp[i][j]:
                    double_list.append((i,k,table_tmp[i][k]*2))
                    table_tmp[i][k] = -1
                    table_tmp[i][j] = 0
                    flag = True
                    break
                else:
                    if j == k+1:
                        break
                    else:
                        table_tmp[i][k+1] = table_tmp[i][j]
                        table_tmp[i][j] = 0
                        flag = True
                        break
    for double in double_list:
        table_tmp[double[0]][double[1]] = double[2]
        reward += double[2]
        double_table[double[0]][double[1]] = 1
    return table_tmp, reward, flag, double_table

def up(state):
    table_tmp = state.copy()
    double_list = []
    reward = 0
    flag = False
    double_table = torch.zeros(size=(4,4))
    for j in range(4):
        for i in range(1, 4):
            if table_tmp[i][j] == 0:
                continue
            for k in range(i-1, -1, -1):
                if table_tmp[k][j] == 0 and k == 0:
                    table_tmp[k][j] = table_tmp[i][j]
                    table_tmp[i][j] = 0
                    flag = True 
                elif table_tmp[k][j] == 0 and k != 0:
                    continue
                elif table_tmp[k][j] == table_tmp[i][j]:
                    double_list.append((k,j,table_tmp[k][j]*2))
                    table_tmp[k][j] = -1
                    table_tmp[i][j] = 0
                    flag = True
                    break
                else:
                    if i == k+1:
                        break
                    else:
                        table_tmp[k+1][j] = table_tmp[i][j]
                        table_tmp[i][j] = 0
                        flag = True
                        break
    for double in double_list:
        table_tmp[double[0]][double[1]] = double[2]
        reward += double[2]
        double_table[double[0]][double[1]] = 1
    return table_tmp, reward, flag, double_table
    
def down(state):
    table_tmp = state.copy()
    double_list = []
    reward = 0
    flag = False
    double_table = torch.zeros(size=(4,4))
    for j in range(4):
        for i in range(2, -1, -1):
            if table_tmp[i][j] == 0:
                continue
            for k in range(i+1, 4):
                if table_tmp[k][j] == 0 and k == 3:
                    table_tmp[k][j] = table_tmp[i][j]
                    table_tmp[i][j] = 0
                    flag = True
                elif table_tmp[k][j] == 0 and k != 3:
                    continue
                elif table_tmp[k][j] == table_tmp[i][j]:
                    double_list.append((k,j,table_tmp[k][j]*2))
                    table_tmp[k][j] = -1
                    table_tmp[i][j] = 0
                    flag = True
                    break
                else:
                    if i == k-1:
                        break
                    else:
                        table_tmp[k-1][j] = table_tmp[i][j]
                        table_tmp[i][j] = 0
                        flag = True
                        break
    for double in double_list:
        table_tmp[double[0]][double[1]] = double[2]
        reward += double[2]
        double_table[double[0]][double[1]] = 1
    return table_tmp, reward, flag, double_table

class NormalActorCritic(nn.Module):
    def __init__(self, layer_num, channel_num, use_bn):
        super(NormalActorCritic, self).__init__()
        self.layer_num = layer_num
        self.channel_num = channel_num
        self.use_bn = use_bn
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.actor_last_conv = nn.Conv2d(channel_num, 4, kernel_size=4, stride=1, padding=0)
        self.critic_last_conv = nn.Conv2d(channel_num, 1, kernel_size=4, stride=1, padding=0)
        self.convs.append(nn.Conv2d(17, channel_num, kernel_size=3, padding=1))
        self.bns.append(nn.BatchNorm2d(channel_num))
        self.flatten = nn.Flatten()
        for _ in range(layer_num-1):
            self.convs.append(nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1))
            self.bns.append(nn.BatchNorm2d(channel_num))
    
    def actor(self, x):
        h = x
        for i in range(self.layer_num):
            h = self.convs[i](h)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
        h = self.actor_last_conv(h)
        return self.flatten(h)

    def critic(self, x):
        h = x
        for i in range(self.layer_num):
            h = self.convs[i](h)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
        h = self.critic_last_conv(h)
        return h
    
    def compute_actor_critic_values(self, x):
        h = x
        for i in range(self.layer_num):
            h = self.convs[i](h)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
        h_actor = self.actor_last_conv(h)
        h_critic = self.critic_last_conv(h)
        return self.flatten(h_actor), h_critic
    
    def forward(self):
        raise NotImplementedError

    def act(self, state, is_training = True):
        actions = [up, right, down, left]
        flags = []
        legal_actions = []
        legal_logits = []
        for action in range(4):
            afterstate, reward, flag, reward_table = actions[action](state)
            flags.append(flag)
        state = to_3d(state).unsqueeze(0).to(device)
        action_logits = self.actor(state).squeeze()
        for a in range(4):
            if flags[a]:
                legal_actions.append(a)
                legal_logits.append(action_logits[a])
        legal_policy = F.softmax(torch.tensor(legal_logits), dim = -1)
        dist = Categorical(legal_policy)
        action = dist.sample() if is_training else torch.argmax(legal_logits)
        action_logprob = dist.log_prob(action)
        action = legal_actions[action]
        return action, action_logprob
    
    def evaluate(self, batch_state, batch_actions):
        actor_logits = self.actor(batch_state)
        action_probs = F.softmax(actor_logits, dim = -1)
        state_values = self.critic(batch_state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(batch_actions)
        return action_logprobs, state_values

class AfterstateActorCritic(nn.Module):
    def __init__(self, layer_num, channel_num, use_bn):
        super(AfterstateActorCritic, self).__init__()
        self.layer_num = layer_num
        self.channel_num = channel_num
        self.use_bn = use_bn
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        self.actor_last_conv = nn.Conv2d(channel_num, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.critic_last_conv = nn.Conv2d(channel_num, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.convs.append(nn.Conv2d(17, channel_num, kernel_size=3, padding=1, bias=False))
        self.bns.append(nn.BatchNorm2d(channel_num))
        for _ in range(layer_num-1):
            self.convs.append(nn.Conv2d(channel_num, channel_num, kernel_size=3, padding=1, bias=False))
            self.bns.append(nn.BatchNorm2d(channel_num))
    
    def actor(self, x):
        h = x
        for i in range(self.layer_num):
            h = self.convs[i](h)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
        h = self.actor_last_conv(h)
        return h.view((-1, 1))
    
    def critic(self, x):
        h = x
        for i in range(self.layer_num):
            h = self.convs[i](h)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
        h = self.critic_last_conv(h)
        return h.view((-1, 1))
    
    def compute_actor_critic_values(self, x):
        h = x
        for i in range(self.layer_num):
            h = self.convs[i](h)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
        h_actor = self.actor_last_conv(h)
        h_critic = self.critic_last_conv(h)
        return h_actor.view((-1,1)), h_critic.view((-1,1))

    def forward(self):
        raise NotImplementedError

    def act(self, state, afterstates_buffer, rewards_buffer, is_training = True):
        actions = [up, right, down, left]
        afterstate_tensors = []
        flags = []
        for action in range(4):
            afterstate, reward, flag, reward_table = actions[action](state)
            afterstate_tensor = to_3d(afterstate)
            afterstate_tensors.append(afterstate_tensor)
            afterstates_buffer.append(afterstate_tensor)
            rewards_buffer.append(reward)
            flags.append(flag)
        afterstate_tensors = torch.stack(afterstate_tensors).to(device)
        action_logits = self.actor(afterstate_tensors).squeeze()
        legal_actions = []
        legal_logits = []
        for a in range(4):
            if flags[a]:
                legal_actions.append(a)
                legal_logits.append(action_logits[a])
        legal_policy = F.softmax(torch.tensor(legal_logits), dim = -1)
        temp_dist = Categorical(legal_policy)
        action = temp_dist.sample() if is_training else torch.argmax(legal_logits)
        action_logprob = temp_dist.log_prob(action)
        action = legal_actions[action]
        return action, action_logprob.detach()

    def evaluate(self, state, action, afterstates, rewards):
        #afterstatesとstateが同じが判定する
        afterstate_actor_values, afterstate_critic_values = self.compute_actor_critic_values(afterstates)
        #afterstate_actor_values = self.actor(afterstates)
        afterstate_actor_values = afterstate_actor_values.view(-1, 4)
        action_probs = F.softmax(afterstate_actor_values, dim=-1)
        dist = Categorical(action_probs)
        entropy = dist.entropy()
        action_logprobs = dist.log_prob(action)

        #state_values = self.critic(state)
        afterstate_critic_values = afterstate_critic_values
        #afterstate_critic_values = afterstate_critic_values + rewards.unsqueeze(1)
        afterstate_critic_values = afterstate_critic_values.view(-1, 4)
        state_values = (action_probs.detach() * afterstate_critic_values).sum(1) #actorの出力する確率で重み付き平均を計算する
        return action_logprobs, state_values, entropy

class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(17, 256, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        raise NotImplementedError
    def act(self, state, is_training = True):
        rand = np.random.random()
        eps = 0.1 if is_training else 0
        if rand < eps:
            action = np.random.choice([0,1,2,3])
        else:
            actions = [up, right, down, left]
            afterstate_tensors = []
            rewards = []
            illegal_actions = []
            for a in range(4):
                afterstate, reward, flag, double_table = actions[a](state)
                afterstate_tensors.append(to_3d(afterstate))
                rewards.append(reward)
                if not flag:
                    illegal_actions.append(a)
            afterstate_tensors = torch.stack(afterstate_tensors).to(device)
            #rewards = torch.tensor(rewards).to(device)
            #afterstate_values = self.model(afterstate_tensors).squeeze() + rewards
            afterstate_values = self.model(afterstate_tensors).squeeze()
            for a in illegal_actions:
                afterstate_values[a] = 0.
            action =  torch.argmax(afterstate_values).item()
        return action
    def evaluate(self, batch_afterstates, batch_next_afterstates): #batch_afterstates, batch_next_afterstatesは通常の配列
        batch_afterstates = torch.stack(batch_afterstates).to(device)
        batch_afterstate_values = self.model(batch_afterstates).squeeze()
        batch_next_afterstate_values = self.compute_next_state_values(batch_next_afterstates)
        return batch_afterstate_values, batch_next_afterstate_values
    
    def compute_next_state_values(self, batch_next_afterstates):
        batch_next_afterstates = batch_next_afterstates[1:]
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_afterstates))).to(device)
        non_final_next_states = torch.stack([s for s in batch_next_afterstates if s is not None]).to(device)
        next_state_values = torch.zeros(len(batch_next_afterstates)).to(device)
        next_state_values[non_final_mask] = torch.squeeze(self.model(non_final_next_states)).to(device)
        return next_state_values.detach()

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.next_n_step_states = []
        self.afterstates = []
        self.logprobs = []
        self.afterstate_rewards = []
        self.rewards = []
        self.is_terminals = []
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.next_states[:]
        del self.next_n_step_states[:]
        del self.afterstates[:]
        del self.logprobs[:]
        del self.afterstate_rewards[:]
        del self.rewards[:]
        del self.is_terminals[:]

def to_3d(state):
    state_copy = state.copy()
    state_tensor = torch.FloatTensor(state_copy)
    a = []
    for i in range(0, 17):
        if i == 0:
            a.append(torch.where(state_tensor==0, 1, 0))
        else:
            a.append(torch.where(state_tensor==2**i, 1, 0))
    return torch.stack(a, dim = 0).type(torch.FloatTensor)

if __name__ == '__main__':
    table = np.array([
        [2,8,4,16],
        [0,0,0,0],
        [0,0,4,128],
        [0,32,4,8]
    ])
    table_tmp, reward, flag = right(table)
    print(table)
    print(table_tmp)
    print(reward)
    print(flag)