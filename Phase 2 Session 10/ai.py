import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque

class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.choice(range(0,len(self.storage)),batch_size,replace=False)
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=state_dim[2],
                      out_channels=8,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(8),
                      nn.Dropout(0.1))  # output_size = 32

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1))  # output_size = 32

        # TRANSITION BLOCK 1

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=0,
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) #output 16

        self.convblock4 = nn.Sequential(
          nn.Conv2d(in_channels=16,
                    out_channels=16,
                    kernel_size=(3, 3),
                    padding=(1,1),
                    bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(16),
                    nn.Dropout(0.1)) #output 16

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) # output 16

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=2,
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) #output 8 

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) #output 8

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) # output 8

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=2,
                      bias=False)) # output 4
        
        self.GAP = nn.AvgPool2d(16)
        self.max_action = max_action

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.GAP(x)
        x = x.view(-1, 16)
        x = self.max_action * torch.tanh(nn.Linear(x,action_dim))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Define first Critic neural network
        # Action_dim angle after GAP as put on same dimension
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=state_dim[2],
                      out_channels=8,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(8),
                      nn.Dropout(0.1))  # output_size = 32

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1))  # output_size = 32

        # TRANSITION BLOCK 1

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=0,
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) #output 16

        self.convblock4 = nn.Sequential(
          nn.Conv2d(in_channels=16,
                    out_channels=16,
                    kernel_size=(3, 3),
                    padding=(1,1),
                    bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(16),
                    nn.Dropout(0.1)) #output 16

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) # output 16

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=2,
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) #output 8 

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) #output 8

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) # output 8

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=2,
                      bias=False)) # output 4
        
        self.GAP1 = nn.AvgPool2d(16)
        self.Linear1 = nn.Linear(17,1)

        # Define the second Critic neural network
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=state_dim[2],
                      out_channels=8,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(8),
                      nn.Dropout(0.1))  # output_size = 32

        # CONVOLUTION BLOCK 1
        self.convblock22 = nn.Sequential(
            nn.Conv2d(in_channels=8,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1))  # output_size = 30

        self.convblock33 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=0,
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) #output 16

        self.convblock44 = nn.Sequential(
          nn.Conv2d(in_channels=16,
                    out_channels=16,
                    kernel_size=(3, 3),
                    padding=(1,1),
                    bias=False),
                    nn.ReLU(),
                    nn.BatchNorm2d(16),
                    nn.Dropout(0.1)) #output 16

        self.convblock55 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) # output 16

        self.convblock66 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=2,
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) #output 8 

        self.convblock77 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) #output 8

        self.convblock88 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      padding=(1,1),
                      bias=False),
                      nn.ReLU(),
                      nn.BatchNorm2d(16),
                      nn.Dropout(0.1)) # output 8

        # OUTPUT BLOCK
        self.convblock99 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=(3, 3),
                      stride=2,
                      bias=False)) # output 4
        
        self.GAP2 = nn.AvgPool2d(16)
        self.Linear2 = nn.Linear(17,1)

    def forward(self, x, u):
        # Forward-Propogation on the first Critic Neural network
        x1 = self.convblock1(x)
        x1 = self.convblock2(x1)
        x1 = self.convblock3(x1)
        x1 = self.convblock4(x1)
        x1 = self.convblock5(x1)
        x1 = self.convblock6(x1)
        x1 = self.convblock7(x1)
        x1 = self.convblock8(x1)
        x1 = self.convblock9(x1)
        x1 = self.GAP1(x1)
        x1 = x1.view(-1, 16)
        x1 = torch.cat([x1, u], 1)
        x1 = self.Linear1(x1, 1)

        # Forward-Propagation on the second Critic Neural Network
        x2 = self.convblock11(x)
        x2 = self.convblock22(x2)
        x2 = self.convblock33(x2)
        x2 = self.convblock44(x2)
        x2 = self.convblock55(x2)
        x2 = self.convblock66(x2)
        x2 = self.convblock77(x2)
        x2 = self.convblock88(x2)
        x2 = self.convblock99(x2)
        x2 = self.GAP2(x2)
        x2 = x2.view(-1, 16)
        x2 = torch.cat([x2, u], 1)
        x2 = self.Linear2(x2, 1)

        return x1, x2

    def Q1(self, x, u):
        x1 = self.convblock1(x)
        x1 = self.convblock2(x1)
        x1 = self.convblock3(x1)
        x1 = self.convblock4(x1)
        x1 = self.convblock5(x1)
        x1 = self.convblock6(x1)
        x1 = self.convblock7(x1)
        x1 = self.convblock8(x1)
        x1 = self.convblock9(x1)
        x1 = self.GAP1(x1)
        x1 = x1.view(-1, 16)
        x1 = torch.cat([x1, u], 1)
        x1 = self.Linear1(x1, 1)
        return x1



# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state):
    # state = torch.Tensor(state.reshape(1, -1)).to(device)
    return self.actor(state).cpu().data.numpy().flatten()  

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      done = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))


def evaluate_policy(policy, eval_episodes=10):
  avg_reward = 0.
  for _ in range(eval_episodes):
    obs = env.reset()
    done = False
    while not done:
      action = policy.select_action(np.array(obs))
      obs, reward, done, _ = env.step(action)
      avg_reward += reward
  avg_reward /= eval_episodes
  print ("---------------------------------------")
  print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
  print ("---------------------------------------")
  return avg_reward

