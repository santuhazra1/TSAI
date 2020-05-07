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
    batch_states, batch_next_states, batch_actions, batch_rewards, batch_distances_curr, batch_distances_nxt, batch_orientations_curr, batch_orientations_nxt, batch_dones = [], [], [], [], [], [], [], [], []
    for i in ind: 
      state, next_state, action, reward, distance_curr, distance_nxt, orientation_curr, orientation_nxt, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_distances_curr.append(np.array(distance_curr, copy=False))
      batch_distances_nxt.append(np.array(distance_nxt, copy=False))
      batch_orientations_curr.append(np.array(orientation_curr, copy=False))
      batch_orientations_nxt.append(np.array(orientation_nxt, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions).reshape(-1, 1), np.array(batch_rewards).reshape(-1, 1), np.array(batch_distances_curr).reshape(-1, 1), np.array(batch_distances_nxt).reshape(-1, 1), np.array(batch_orientations_curr).reshape(-1, 1), np.array(batch_orientations_nxt).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, distance_flag, orientation_flag):
        super(Actor, self).__init__()
        
        self.distance_flag = distance_flag
        self.orientation_flag = orientation_flag

        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=state_dim[0], out_channels=8, kernel_size=(3, 3), padding=(1,1), bias=False), nn.ReLU(), nn.BatchNorm2d(8), nn.Dropout(0.1))  # output_size = 32

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1))  # output_size = 32

        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),stride=2,padding=0,bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 16

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 16

        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1)) # output 16

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),stride=2,bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 8 

        self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 8

        self.GAP = nn.AvgPool2d(7,7)

        self.max_action = max_action
        if self.distance_flag == True & self.orientation_flag == True:
          self.LinearA1 = nn.Linear(19,200)
          self.LinearA2 = nn.Linear(200,100)
          self.LinearA3 = nn.Linear(100,1)
        else:
          self.LinearA = nn.Linear(16,1)

    def forward(self, x, d, o):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.GAP(x)
        x = x.view(-1, 16)

        if self.distance_flag == True & self.orientation_flag == True:
          x1 =  torch.cat([x, d, o, -o], 1)
          x1 =  F.relu(self.LinearA1(x1))
          x1 =  F.relu(self.LinearA2(x1))
          x1 =  self.LinearA3(x1)
        else:
          x1 = self.LinearA(x)

        x1 = self.max_action * torch.tanh(x1)
        return x1


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, distance_flag, orientation_flag):
        super(Critic, self).__init__()
        # First Critic network
        self.distance_flag = distance_flag
        self.orientation_flag = orientation_flag

        self.convblock1 = nn.Sequential(nn.Conv2d(in_channels=state_dim[0],out_channels=8,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(),nn.BatchNorm2d(8),nn.Dropout(0.1))  # output_size = 32

        self.convblock2 = nn.Sequential(nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1))  # output_size = 32

        self.convblock3 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),stride=2,padding=0,bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 16

        self.convblock4 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),padding=(1,1), bias=False),nn.ReLU(),nn.BatchNorm2d(16), nn.Dropout(0.1)) #output 16

        self.convblock5 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1)) # output 16

        self.convblock6 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),stride=2,bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 8 

        self.convblock7 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 8

        self.GAP1 = nn.AvgPool2d(7,7)

        if self.distance_flag == True & self.orientation_flag == True:
          self.LinearC11 = nn.Linear(20, 200)
          self.LinearC12 = nn.Linear(200, 100)
          self.LinearC13 = nn.Linear(100, 1)
        else:
          self.LinearC11 = nn.Linear(17, 200)
          self.LinearC12 = nn.Linear(200, 100)
          self.LinearC13 = nn.Linear(100, 1)


        # Second Critic network
        self.convblock11 = nn.Sequential(nn.Conv2d(in_channels=state_dim[0],out_channels=8,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(),nn.BatchNorm2d(8),nn.Dropout(0.1))  # output_size = 32

        self.convblock22 = nn.Sequential(nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1))  # output_size = 30

        self.convblock33 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),stride=2,padding=0,bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 16

        self.convblock44 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(), nn.BatchNorm2d(16), nn.Dropout(0.1)) #output 16

        self.convblock55 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.ReLU(),nn.BatchNorm2d(16),nn.Dropout(0.1)) # output 16

        self.convblock66 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),stride=2,bias=False),nn.ReLU(), nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 8 

        self.convblock77 = nn.Sequential(nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3, 3),padding=(1,1),bias=False),nn.BatchNorm2d(16),nn.Dropout(0.1)) #output 8

        self.GAP2 = nn.AvgPool2d(7,7)
        if self.distance_flag == True & self.orientation_flag == True:        
          self.LinearC21 = nn.Linear(20, 200)
          self.LinearC22 = nn.Linear(200, 100)
          self.LinearC23 = nn.Linear(100, 1)
        else:
          self.LinearC21 = nn.Linear(17, 200)
          self.LinearC22 = nn.Linear(200, 100)
          self.LinearC23 = nn.Linear(100, 1)          

    def forward(self, x, u, d, o):
        # Forward-Propogation on the first Critic Neural network
        x1 = self.convblock1(x)
        x1 = self.convblock2(x1)
        x1 = self.convblock3(x1)
        x1 = self.convblock4(x1)
        x1 = self.convblock5(x1)
        x1 = self.convblock6(x1)
        x1 = self.convblock7(x1)
        x1 = self.GAP1(x1)
        x1 = x1.view(-1, 16)
        if self.distance_flag == True & self.orientation_flag == True:         
          x1 = torch.cat([x1, u, d, o, -o], 1)
          x1 = F.relu(self.LinearC11(x1))
          x1 = F.relu(self.LinearC12(x1))
          x1 = self.LinearC13(x1)
        else:
          x1 = torch.cat([x1, u], 1)
          x1 = F.relu(self.LinearC11(x1))
          x1 = F.relu(self.LinearC12(x1))
          x1 = self.LinearC13(x1)

        # Forward-Propagation on the second Critic Neural Network
        x2 = self.convblock11(x)
        x2 = self.convblock22(x2)
        x2 = self.convblock33(x2)
        x2 = self.convblock44(x2)
        x2 = self.convblock55(x2)
        x2 = self.convblock66(x2)
        x2 = self.convblock77(x2)
        x2 = self.GAP2(x2)
        x2 = x2.view(-1, 16)
        if self.distance_flag == True & self.orientation_flag == True:          
          x2 = torch.cat([x2, u, d, o, -o], 1)
          x2 = F.relu(self.LinearC21(x2))
          x2 = F.relu(self.LinearC22(x2))
          x2 = self.LinearC23(x2)
        else:
          x2 = torch.cat([x2, u], 1)
          x2 = F.relu(self.LinearC21(x2))
          x2 = F.relu(self.LinearC22(x2))
          x2 = self.LinearC23(x2)          

        return x1, x2

    def Q1(self, x, u, d, o):
        x1 = self.convblock1(x)
        x1 = self.convblock2(x1)
        x1 = self.convblock3(x1)
        x1 = self.convblock4(x1)
        x1 = self.convblock5(x1)
        x1 = self.convblock6(x1)
        x1 = self.convblock7(x1)
        x1 = self.GAP1(x1)
        x1 = x1.view(-1, 16)
        if self.distance_flag == True & self.orientation_flag == True:         
          x1 = torch.cat([x1, u, d, o, -o], 1)
          x1 = F.relu(self.LinearC11(x1))
          x1 = F.relu(self.LinearC12(x1))
          x1 = self.LinearC13(x1)
        else:
          x1 = torch.cat([x1, u], 1)
          x1 = F.relu(self.LinearC11(x1))
          x1 = F.relu(self.LinearC12(x1))
          x1 = self.LinearC13(x1)
        return x1



# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action, distance_flag = True, orientation_flag = True):
    self.actor = Actor(state_dim, action_dim, max_action, distance_flag, orientation_flag).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action, distance_flag, orientation_flag).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    self.critic = Critic(state_dim, action_dim, distance_flag, orientation_flag).to(device)
    self.critic_target = Critic(state_dim, action_dim, distance_flag, orientation_flag).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    self.max_action = max_action

  def select_action(self, state, distance, orientation):
    state = torch.Tensor(state).to(device)
    distance = torch.Tensor(distance).to(device).view(1,1)
    orientation = torch.Tensor(orientation).to(device).view(1,1)
    return self.actor(state, distance, orientation).cpu().data.numpy().flatten()  

  def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states, batch_next_states, batch_actions, batch_rewards, batch_distances_curr, batch_distances_nxt, batch_orientations_curr, batch_orientations_nxt, batch_dones = replay_buffer.sample(batch_size)
      state = torch.Tensor(batch_states).to(device)
      next_state = torch.Tensor(batch_next_states).to(device)
      action = torch.Tensor(batch_actions).to(device)
      reward = torch.Tensor(batch_rewards).to(device)
      distance_curr = torch.Tensor(batch_distances_curr).to(device)
      distance_nxt = torch.Tensor(batch_distances_nxt).to(device)
      orientation_curr = torch.Tensor(batch_orientations_curr).to(device)
      orientation_nxt = torch.Tensor(batch_orientations_nxt).to(device)
      done = torch.Tensor(batch_dones).to(device)

      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state, distance_nxt, orientation_nxt)
      next_action = next_action.view(16)
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      noise = noise.view(16)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      next_action = next_action.view(16, -1)

      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action, distance_nxt, orientation_nxt)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      action = action.view(16, -1)
      current_Q1, current_Q2 = self.critic(state, action, distance_curr, orientation_curr)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state, distance_curr, orientation_curr), distance_curr, orientation_curr).mean()
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




