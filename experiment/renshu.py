import numpy as np
import gym
import torch
import sys
import os
import argparse

sys.path.append("..")
sys.path.append("../agent")
import gym_2048
import ppo
import utils
from ppo import AfterstatePPO, NormalPPO
import matplotlib.pyplot as plt
from collections import defaultdict

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

agent = AfterstatePPO(4, 64, True, 0.002, 0.99, 0.99, 30, 0.2)
agent.model.load_state_dict(torch.load("ppo.pth", map_location=torch.device('cpu')))
agent.model_old.load_state_dict(torch.load("ppo.pth", map_location=torch.device('cpu')))
agent.model_old.eval()

env = gym.make("2048-v0")
d = defaultdict(int)
array = []
for i in range(100):
    state = env.reset()
    score = 0
    while True:
        action = agent.select_action(state, is_training=False)
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            max_number = state.max()
            d[max_number] += 1
            break
    array.append(score)
array = np.array(array)
print(d)
print(np.mean(array))