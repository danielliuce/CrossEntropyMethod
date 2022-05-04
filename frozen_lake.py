import gym
import gym.spaces
import gym.wrappers
import gym.envs.toy_text.frozen_lake
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

import random

#Some pitfalls of cross entropy method are present and are addressed below

HIDDEN_SIZE = 128
BATCH_SIZE = 100 #Large batches of played episodes to get more successful episodes
PERCENTILE = 30
GAMMA = 0.9 #Discount factor applied to reward for episode to depend on episode length

#Keep elite episodes for a longer time since they are so rare
#Decrease the learning rate to give network time to average more training samples
#Increase training time, more itreations

#information about the environment 
e = gym.make("FrozenLake-v0")

print(e.observation_space)
print(e.action_space)
print(e.reset())
e.render()

#Discrete environment, number from 0 to 15 
#Action space discrete, 0 to 3

class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (env.observation_space.n, ), dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation]=1.0
        return res

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs]) #convert a 4x1 into tensor of 1x4
        act_probs_v = sm(net(obs_v)) #feed to softmax function for probability distribution
        act_probs = act_probs_v.data.numpy()[0] #returns tensors which track gradients, unpack them into a NumPy array
      
        #we have probability of actions
        #use this distribution to obtain actual action at random
        #obtain next observation, reward, indcitaion episode is ending, and extra info
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, extra_info = env.step(action)

        #Step number 2, calculate total award for every espidoe
        #Accumulate total award
        #List of episode steps extended with an observation, action pair
        #Careful! we save observation that was used to choose action, not next_obs
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        #Reset everything after appending finalized episdoe to batch with total award and steps
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            #Reached desired count of episodes, return to caller using yield
            if(len(batch) == batch_size):
                yield batch
                batch = [] #reset for next batch
        #assign an observation obtained from environment to curr observation
        obs = next_obs

#Change this function to account for pitfalls
def filter_batch(batch, percentile):
    disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)
    train_obs = []
    train_act = []
    elite_batch = []
    
    #example maps to first element of zipped together while discounted_reward maps to second
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)
    return elite_batch, train_obs, train_act, reward_bound

if __name__ == "__main__":
    random.seed(12345)
    #Speeding it up
    env = gym.envs.toy_text.frozen_lake.FrozenLakeEnv(is_slippery=False)
    env = gym.wrappers.TimeLimit(env, max_episode_steps = 100)
    env = DiscreteOneHotWrapper(env)

    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)
    writer = SummaryWriter(comment = "-frozenlake-nonslippery")

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env,net,BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_bound = filter_batch(full_batch+batch, PERCENTILE)

        if not full_batch:
            continue
        obs_v = torch.FloatTensor(obs)
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]

        #Classic training algorithm
        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()

        
        #Keeping track of progress
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss_v.item(), reward_mean, reward_bound))
        writer.add_scalar("loss", loss_v.item(), iter_no)
        writer.add_scalar("reward_bound", reward_bound, iter_no)
        writer.add_scalar("reward_mean", reward_mean, iter_no)
        if reward_mean > 0.8: 
            print("Solved!")
            break
    writer.close()
