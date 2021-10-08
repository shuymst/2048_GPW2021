# This program examines the ability of afterstate td agent when reward is not score but episode length

import numpy as np
import gym
import torch
import sys
import os
import argparse

sys.path.append("..")
sys.path.append("../agent")
import gym_2048
from utils import to_3d
from td import TDAgent
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

#################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--max_training_episodes', default=30000, type=int)
parser.add_argument('--lr', default=0.0003, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--seed', default=123, type=int)

args = parser.parse_args()

max_training_episodes = args.max_training_episodes
lr = args.lr
gamma = args.gamma
seed = args.seed

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
######################################################################

env = gym.make("2048-v0")

program_dir = "program3"
figure_dir = "program3/figure"
model_dir = "program3/model_params"
score_dir = "program3/score"

if not os.path.exists(program_dir):
    os.makedirs(program_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

score_history = []
score_history_evaluate = []

td_agent = TDAgent(gamma, lr)

for i_episode in range(1, max_training_episodes+1):
    state = env.reset()
    score = 0
    while True:
        action = td_agent.select_action(state)
        state, reward, done, afterstate = env.step(action)
        score += reward
        reward = 1.0
        if done:
            td_agent.buffer_next_rewards.append(0)
            td_agent.buffer_next_afterstates.append(None)
        else:
            td_agent.buffer_afterstates.append(to_3d(afterstate))
            td_agent.buffer_next_afterstates.append(to_3d(afterstate))
            td_agent.buffer_next_rewards.append(reward)
        if done:
            print(f"episode {i_episode}: {score}")
            break 
    td_agent.update()
    score_history.append(score)
    if i_episode % 100 == 0:
        evaluate_scores = []
        for _ in range(10):
            state = env.reset()
            score = 0
            while True:
                action = td_agent.select_action(state, is_training = False)
                state, reward, done, _ = env.step(action)
                score += reward
                if done:
                    break
            evaluate_scores.append(score)
        evaluate_scores = np.array(evaluate_scores)
        score_history_evaluate.append(evaluate_scores.mean())

torch.save(td_agent.model.state_dict(), "./program3/model_params/td_with_step_rewardpth")

fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "episode", ylabel='total rewards')
ax.set_title("Afterstate TD(trained with step reward)")
score_history = np.array(score_history)
score_history_evaluate = np.array(score_history_evaluate)
np.save('program3/score/score_history', score_history)
np.save('program3/score/score_history_evaluate', score_history_evaluate)
num = 100
b = np.ones(num) / num
moving_average = np.convolve(score_history, b, mode="same")
#ax.plot(range(1, max_training_episodes+1), moving_average, color = "blue")
ax.plot(range(100, max_training_episodes+1, 100), score_history_evaluate, color="blue")
ax.plot(range(1, max_training_episodes+1), score_history, color = "blue", alpha = 0.2)
plt.savefig(os.path.join(figure_dir, 'Afterstate TD with step reward.png'))