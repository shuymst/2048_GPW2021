#You can compare the influence of type of reward

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
from ppo import AfterstatePPO
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

#################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--max_training_episodes', default=30000, type=int)
parser.add_argument('--update_timestep', default=4000, type=int)
parser.add_argument('--layer_num', default=4, type=int)
parser.add_argument('--channel_num', default=64, type=int)
parser.add_argument('--use_bn', default=True, type=bool)
parser.add_argument('--lr', default=0.002, type=float)
parser.add_argument('--K_epochs', default=30, type=int)
parser.add_argument('--eps_clip', default=0.2, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--lambd', default=0.99, type=float)
parser.add_argument('--seed', default=123, type=int)

args = parser.parse_args()

max_training_episodes = args.max_training_episodes
update_timestep = args.update_timestep
layer_num = args.layer_num
channel_num = args.channel_num
use_bn = args.use_bn
lr = args.lr
K_epochs = args.K_epochs
eps_clip = args.eps_clip
gamma = args.gamma
lambd = args.lambd
seed = args.seed

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
######################################################################

env = gym.make("2048-v0")

program_dir = "program2"
figure_dir = "program2/figure"
model_dir = "program2/model_params"
score_dir = "program2/score"

if not os.path.exists(program_dir):
    os.makedirs(program_dir)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)

score_history_normal_reward = []
score_history_normal_reward_evaluate = []
score_history_step_reward = []
score_history_step_reward_evaluate = []

for i in range(2):
    #i == 0 reward:score
    #i == 1 reward:step 
    ppo_agent = AfterstatePPO(layer_num, channel_num, use_bn, lr, gamma, lambd, K_epochs, eps_clip)

    time_step = 0
    i_episode = 0
    update_count = 0
    update_game_count = 0

    for i_episode in range(1, max_training_episodes + 1):
        state = env.reset()
        score = 0
        in_episode_time = 0
        while True:
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)
            score += reward
            if i == 1:
                reward = 1.0
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            in_episode_time += 1
            time_step += 1
            update_count += 1
            if done:
                update_game_count += 1
                if i == 0:
                    score_history_normal_reward.append(score)
                else:
                    score_history_step_reward.append(score)
                print(f"episode {i_episode}: {score}")
                break 
        if update_count >= update_timestep and update_game_count >= 10:
            ppo_agent.update()
            update_count = 0
            update_game_count = 0
        if i_episode % 100 == 0:
            evaluate_scores = []
            for _ in range(10):
                state = env.reset()
                score = 0
                while True:
                    action = ppo_agent.select_action(state, is_training = False)
                    state, reward, done, _ = env.step(action)
                    score += reward
                    if done:
                        break
                evaluate_scores.append(score)
            evaluate_scores = np.array(evaluate_scores)
            if i == 0:
                score_history_normal_reward_evaluate.append(evaluate_scores.mean())
            else:
                score_history_step_reward_evaluate.append(evaluate_scores.mean())
    if i == 0:
        torch.save(ppo_agent.model.state_dict(), "./program2/model_params/normal_reward.pth")
    else:
        torch.save(ppo_agent.model.state_dict(), "./program2/model_params/step_reward.pth")

fig = plt.figure()
ax = fig.add_subplot(111, xlabel = "episode", ylabel='total rewards')
ax.set_title("normal reward vs step reward")
score_history_normal_reward = np.array(score_history_normal_reward)
score_history_step_reward = np.array(score_history_step_reward)
score_history_normal_reward_evaluate = np.array(score_history_normal_reward_evaluate)
score_history_step_reward_evaluate = np.array(score_history_step_reward_evaluate)
np.save('program2/score/score_history_normal_reward', score_history_normal_reward)
np.save('program2/score/score_history_step_reward', score_history_step_reward)
np.save('program2/score/score_history_normal_reward_evaluate', score_history_normal_reward_evaluate)
np.save('program2/score/score_history_step_reward_evaluate', score_history_step_reward_evaluate)
num = 100
b = np.ones(num) / num
moving_average_normal_reward = np.convolve(score_history_normal_reward, b, mode="same")
moving_average_step_reward = np.convolve(score_history_step_reward, b, mode="same")
#ax.plot(range(1, max_training_episodes+1), moving_average_normal_reward, color="blue")
ax.plot(range(100, max_training_episodes+1, 100), score_history_normal_reward_evaluate, color="blue")
ax.plot(range(100, max_training_episodes+1, 100), score_history_step_reward_evaluate, color="red")
ax.plot(range(1, max_training_episodes+1), score_history_normal_reward, color="blue", alpha=0.2)
#ax.plot(range(1, max_training_episodes+1), moving_average_step_reward, color="red")
ax.plot(range(1, max_training_episodes+1), score_history_step_reward, color="red", alpha=0.2)
ax.legend(["normal reward", "step reward"])
plt.savefig(os.path.join(figure_dir, 'compare_normal_reward_and_step_reward.png'))