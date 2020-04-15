#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
installing dependencies
"""
get_ipython().system('apt-get -qq -y install libnvtoolsext1 > /dev/null')
get_ipython().system('ln -snf /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so.8.0 /usr/lib/x86_64-linux-gnu/libnvrtc-builtins.so')
get_ipython().system('apt-get -qq -y install xvfb freeglut3-dev ffmpeg> /dev/null')
get_ipython().system('pip -q install gym')
get_ipython().system('pip -q install pyglet')
get_ipython().system('pip -q install pyopengl')
get_ipython().system('pip -q install pyvirtualdisplay')


# In[ ]:


"""
Imports
"""

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import deque
import random
from gym import wrappers

from pyvirtualdisplay import Display
display = Display(visible=0, size=(1024, 768))
display.start()
import os
os.environ["DISPLAY"] = ":" + str(display.display) + "." + str(display.screen)

import matplotlib.animation
import numpy as np
from IPython.display import HTML


# In[ ]:


import torch
import torchvision

import torch.utils.tensorboard as tb

from PIL import Image

from torch.utils import data 

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from datetime import datetime
import glob, os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# # Homework 6 - Imitation and Reinforcement Learning

# ### Getting to know OpenAI Gym. (TODO)

# 
# We will be using the OpenAI Gym as our environment -- **we strongly recommend looking over the ["Getting Started" documentation](https://gym.openai.com/docs/) .**
# 
# 
# > A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.
# 
# 
# ![](https://user-images.githubusercontent.com/8510097/31701297-3ebf291c-b384-11e7-8289-24f1d392fb48.PNG)
# 
# 
# >The goal position is 0.5, the location of the flag on top of the hill.
# 
# >Reward: -1 for each time step, until the goal position of 0.5 is reached.
# 
# >Initialization: Random position from -0.6 to -0.4 with no velocity.
# 
# >Episode ends when you reach 0.5 position, or if 200 timesteps are reached. (So failure to reach the flag will result in a reward of -200).
# 

# In[ ]:


class ResizeObservation(gym.Wrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.env = env
        self.shape = tuple(shape)

    def render(self):

        from PIL import Image
        obs = self.env.render(mode = 'rgb_array')
        im = Image.fromarray(np.uint8(obs))
        im = im.resize(self.shape)
        return np.asarray(im)



def dummy_policy(env, num_episodes):

  '''
  TODO: Fill in this function 

  Functionality: This should be executing a random policy sampled from the action space of the environment for num_episodes long
                  and should be returning the mean reward over those episodes and the frames of the rendering recorded on the last episode 
  
  Input: env, the MountainCar environment object 
         num_episodes, int, the total number of episodes you want to run this for 

  Returns: mean_reward, float, which is the mean_reward over num_episodes 
           frames, a list, which should contain elements of image dimensions (i.e RGB, with size that you specify), should have a length of the last episode that you record.

  '''

  return mean_reward, frames

resize_observation_shape = 100
env = gym.make('MountainCar-v0')
env = ResizeObservation(env, resize_observation_shape)

rew, frames = dummy_policy(env, 10)

#### Video plotting code ######################
plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
patch = plt.imshow(frames[0])
plt.axis('off')
animate = lambda i: patch.set_data(frames[i])
ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval = 50)
HTML(ani.to_jshtml())


# ## Expert Reinforcement Learning Code - Q-Learning
# 
# 
# You are given the code for training a traditional Q-learning based agent. Please go through this code.

# ### Supporting functions

# In[ ]:


def discretize(state, discretization, env):

    env_minimum = env.observation_space.low
    state_adj = (state - env_minimum)*discretization
    discretized_state = np.round(state_adj, 0).astype(int)

    return discretized_state


def choose_action(epsilon, Q, state, env):
    """    
    Choose an action according to an epsilon greedy strategy.
    Args:
        epsilon (float): the probability of choosing a random action
        Q (np.array): The Q value matrix, here it is 3D for the two observation states and action states
        state (Box(2,)): the observation state, here it is [position, velocity]
        env: the RL environment 
        
    Returns:
        action (int): the chosen action
    """
    action = 0
    if np.random.random() < 1 - epsilon:
        action = np.argmax(Q[state[0], state[1]]) 
    else:
        action = np.random.randint(0, env.action_space.n)
  
    return action


def update_epsilon(epsilon, decay_rate):
    """
    Decay epsilon by the specified rate.
    
    Args:
        epsilon (float): the probability of choosing a random action
        decay_rate (float): the decay rate (between 0 and 1) to scale epsilon by
        
    Returns:
        updated epsilon
    """
  
    epsilon *= decay_rate

    return epsilon


def update_Q(Q, state_disc, next_state_disc, action, discount, learning_rate, reward, terminal):
    """
    
    Update Q values following the Q-learning update rule. 
    
    Be sure to handle the terminal state case.
    
    Args:
        Q (np.array): The Q value matrix, here it is 3D for the two observation states and action states
        state_disc (np.array): the discretized version of the current observation state [position, velocity]
        next_state_disc (np.array): the discretized version of the next observation state [position, velocity]
        action (int): the chosen action
        discount (float): the discount factor, may be referred to as gamma
        learning_rate (float): the learning rate, may be referred to as alpha
        reward (float): the current (immediate) reward
        terminal (bool): flag for whether the state is terminal
        
    Returns:
        Q, with the [state_disc[0], state_disc[1], action] entry updated.
    """    
    if terminal:        
        Q[state_disc[0], state_disc[1], action] = reward

    # Adjust Q value for current state
    else:
        delta = learning_rate*(reward + discount*np.max(Q[next_state_disc[0], next_state_disc[1]]) - Q[state_disc[0], state_disc[1],action])
        Q[state_disc[0], state_disc[1],action] += delta
  
    return Q


# #### Wrapper for Rendering the Environment

# In[ ]:


class ResizeObservation(gym.Wrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.env = env
        self.shape = tuple(shape)

    def render(self):

        from PIL import Image
        obs = self.env.render(mode = 'rgb_array')
        im = Image.fromarray(np.uint8(obs))
        im = im.resize(self.shape)
        return np.asarray(im)


# ### Main Q-learning Loop

# In[ ]:


def Qlearning(Q, discretization, env, learning_rate, discount, epsilon, decay_rate, max_episodes=5000):
    """
    
    The main Q-learning function, utilizing the functions implemented above.
          
    """
    reward_list = []
    position_list = []
    success_list = []
    success = 0 # count of number of successes reached 
    frames = []
  
    for i in range(max_episodes):
        # Initialize parameters
        done = False # indicates whether the episode is done
        terminal = False # indicates whether the episode is done AND the car has reached the flag (>=0.5 position)
        tot_reward = 0 # sum of total reward over a single
        state = env.reset() # initial environment state
        state_disc = discretize(state,discretization,env)

        while done != True:                 
            # Determine next action 
            action = choose_action(epsilon, Q, state_disc, env)                                      
            # Get next_state, reward, and done using env.step(), see http://gym.openai.com/docs/#environments for reference
            if i==1 or i==(max_episodes-1):
              frames.append(env.render())
            next_state, reward, done, _ = env.step(action) 
            # Discretize next state 
            next_state_disc = discretize(next_state,discretization,env)
            # Update terminal
            terminal = done and next_state[0]>=0.5
            # Update Q
            Q = update_Q(Q,state_disc,next_state_disc,action,discount,learning_rate, reward, terminal)  
            # Update tot_reward, state_disc, and success (if applicable)
            tot_reward += reward
            state_disc = next_state_disc

            if terminal: success +=1 
            
        epsilon = update_epsilon(epsilon, decay_rate) #Update level of epsilon using update_epsilon()

        # Track rewards
        reward_list.append(tot_reward)
        position_list.append(next_state[0])
        success_list.append(success/(i+1))

        if (i+1) % 100 == 0:
            print('Episode: ', i+1, 'Average Reward over 100 Episodes: ',np.mean(reward_list))
            reward_list = []
                
    env.close()
    
    return Q, position_list, success_list, frames


# ### Define Params and Launch Q-learning

# In[ ]:


# Initialize Mountain Car Environment
env = gym.make('MountainCar-v0')

env = ResizeObservation(env,100) #Resize observations

env.seed(42)
np.random.seed(42)
env.reset()

# Parameters    
learning_rate = 0.2 
discount = 0.9
epsilon = 0.8 
decay_rate = 0.95
max_episodes = 5000
discretization = np.array([10,100])


#InitQ
num_states = (env.observation_space.high - env.observation_space.low)*discretization
#Size of discretized state space 
num_states = np.round(num_states, 0).astype(int) + 1
# Initialize Q table
Q = np.random.uniform(low = -1, 
                      high = 1, 
                      size = (num_states[0], num_states[1], env.action_space.n))

# Run Q Learning by calling your Qlearning() function
Q, position, successes, frames = Qlearning(Q, discretization, env, learning_rate, discount, epsilon, decay_rate, max_episodes)

np.save('./expert_Q.npy',Q) #Save the expert


# ### Visualization

# #### Plotting

# In[ ]:


import pandas as pd 

plt.plot(successes)
plt.xlabel('Episode')
plt.ylabel('% of Episodes with Success')
plt.title('% Successes')
plt.show()
plt.close()

p = pd.Series(position)
ma = p.rolling(3).mean()
plt.plot(p, alpha=0.8)
plt.plot(ma)
plt.xlabel('Episode')
plt.ylabel('Position')
plt.title('Car Final Position')
plt.show()


# #### Agent's Video

# In[ ]:


#### Video plotting code #####################
deep_frames = []
for obs in frames:
  im = Image.fromarray(np.uint8(obs))
  im = im.resize((600,400))
  deep_frames.append(np.asarray(im))

plt.figure(figsize=(deep_frames[0].shape[1] / 72.0, deep_frames[0].shape[0] / 72.0), dpi = 72)
patch = plt.imshow(deep_frames[0])
plt.axis('off')
animate = lambda i: patch.set_data(deep_frames[i])
ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(deep_frames), interval = 50)
HTML(ani.to_jshtml())


# ### Generate Expert Trajectories (TODO):
# 
# Using the Q-learning agent above, please complete this block of code to generate expert trajectories

# #### Get actions from expert for a specific observation

# In[ ]:



def get_expert_action(Q,discretization,env,state):
  '''
  TODO: Implement this function

  NOTE: YOU WILL BE USING THIS FUNCTION FOR THE IMITATION LEARNING SECTION AS WELL 

  Functionality: For a given state, returns the action that the expert would take 

  Input: Q value , numpy array
         the discretization 
         env , the environment
         state, (Box(2,)): the observation space state, here it is [position, velocity]

  Returns: action, has dimensions of the action space and type of action space
  '''

  return action


# #### Generate Expert Trajectory

# In[ ]:


def generate_expert_trajectories(Q, discretization, env, num_episodes=150, data_path='./data'):

  '''
  TODO: Implement this function

  Functionality: Execute Expert Trajectories and Save them under the folder of data_path/

  Input: Q value , numpy array
         the discretization 
         env , the environment
         num_episodes, int, which is used to denote number of expert trajectories to store 
         
  Returns: total_samples, int, which denotes the total number of samples that were stored
  '''


  episode_dict['observations'] = episode_observations
  episode_dict['actions'] = episode_actions
  import os
  if not os.path.exists(data_path):
     os.makedirs(data_path)
  np.savez_compressed(data_path+'/episode_number'+'_'+str(i)+'.npz',**episode_dict) #where i can be the episode number that you save

  return total_samples 


# #### Launch code for generating trajectories

# In[ ]:


num_episodes = 100
data_path = './data'


total_samples = generate_expert_trajectories(Q,discretization,env,num_episodes,save_images, data_path) ## Generate trajectories. Use Q, discretization and env by running the previous section

print('---------Total Samples Recorded were --------', total_samples)


# ## Imitation Learning
# 
# Using the trajectories that you collected from the expert above, you will work on imitation learning agents in the code sections below

# ### Working with Data (TODO)
# 
# 
# 

# #### Loading Initial Expert Data

# In[ ]:


def load_initial_data(args):
  '''
  TODO: Fill this function

  Functionality: Reads data from directory and converts them into numpy arrays of observations and actions

  Input arguments: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
  
  Returns: training_observations: numpy array, of shape (B,dim_of_observation), where B is total number of samples that you select
           training_actions: numpy array, of shape (B,dim_of_action), where B is total number of samples that you select

  '''

  return training_observations, training_actions


# #### Convert numpy arrays to a Dataloader

# In[ ]:



def load_dataset(args, observations, actions, batch_size=64, data_transforms=None, num_workers=0):
  '''
  TODO: Fill this function fully 

  Functionality: Converts numpy arrays to dataloaders. 
  
  Inputs: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
          observations, numpy array, of shape (B,dim_of_observation), where B is number of samples 
          actions, numpy array, of shape (B,dim_of_action), where B is number of samples 
          batch_size, int, which you can play around with, but is set to 64 by default. 
          data_transforms, whatever transformations you want to make to your data.

  Returns: dataloader  
          

  '''

  return dataloader


# #### Process Individual Observations

# In[ ]:


def process_individual_observation(args,observation):
  '''
  TODO: Fill this function fully 

  Functionality: Converts individual observations according to the pre-processing that you want  
  
  Inputs: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
          observations, shape (dim_of_observation)

  Returns: data, processed observation that can be fed into the model
  '''

  return data


# ### Defining Networks (TODO)

# #### Define your network for working from States

# In[ ]:


class StatesNetwork(nn.Module):
  '''
  TODO: Implement this class
  '''
    def __init__(self, env):
        """
        Your code here
        """
    
    def forward(self, x):    
        """
        Your code here
        @x: torch.Tensor((B,dim_of_observation))
        @return: torch.Tensor((B,dim_of_actions))
        """

        return forward_pass


# ### Training the model (TODO)

# In[ ]:


def train_model(args):

  '''
  TODO: Fill in the entire train function

  Functionality: Trains the model. How you train this is upto you. 

  Input: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 

  Returns: The trained model 

  '''

    return model


# ### DAgger (TODO)

# #### Get the expert trajectory for imitating agent's observations

# In[ ]:


def execute_dagger(args):

    '''
  TODO: Implement this function

  Functionality: Collect expert labels for the observations seen by the imitation learning agent 
  
  Input: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
         
  Returns: imitation_observations, a numpy array that has dimensions of (episode_length,dim_of_observation)
           expert_actions, a numpy array that has dimensions of (episode_length,dim_of_action)
  '''

          
    return imitation_observations, expert_actions


# #### Aggregate new rollout to the full dataset

# In[ ]:


def aggregate_dataset(training_observations, training_actions, imitation_states, expert_actions):

    '''
  TODO: Implement this function

  Functionality: Adds new expert labeled rollout to the overall dataset

  Input: training_observations, a numpy array that has dimensions of (dataset_size,dim_of_observation)
         training_actions, a numpy array that has dimensions of (dataset_size,dim_of_action)
         imitation_observations, a numpy array that has dimensions of (episode_length,dim_of_observation)
         expert_actions, a numpy array that has dimensions of (episode_length,dim_of_action)

  Returns: training_observations, a numpy array that has dimensions of (updated_dataset_size,dim_of_observation)
           training_actions, a numpy array that has dimensions of (updated_dataset_size,dim_of_action)
  '''


  return training_observations, training_actions 


# ### Utility 
# 
# 

# #### Code for prediction of the network and calculating the accuracy 

# In[ ]:


import numpy as np
from torchvision.transforms import functional as TF

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

def predict(model, inputs, device='cpu'):
    inputs = inputs.to(device)
    logits = model(inputs)
    return F.softmax(logits, -1)


# #### Wrapper for Rendering the environment 
# 
# Same code that was used in the Q-learning agent

# In[ ]:


class ResizeObservation(gym.Wrapper):
    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape
        self.env = env
        self.shape = tuple(shape)

    def render(self):

        from PIL import Image
        obs = self.env.render(mode = 'rgb_array')
        im = Image.fromarray(np.uint8(obs))
        im = im.resize(self.shape)
        return np.asarray(im)


# ### Test model performance

# In[ ]:


def test_model(args, record_frames=False):

    '''
  Functionality: Should take your model and run it for a complete episode (model should either not finish the game in 200 steps or finish the game). Record stats

  Input: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 
         record_frames, Boolean. Denotes if you want to record frames to display them as video.

  Returns: final_position, The final position of the car when the episode ended
           success, Boolean, denotes if the episode was a success or not
           frames, a list of frames that have been rendered throughout the episode. Should have a length of the total episode length
           episode_reward, float, denotes the total rewards obtained while executing this episode
  '''

    frames = []
    env = args.env 
    model = args.model
    state = env.reset()

    model.eval()

    episode_reward = 0

    success = False
    done = False 

    while not done:

        observation = state

        data = preprocess_individual_observation(args,observation)
        logit = model(data)
        action = torch.argmax(logit).item()

        if record_frames: #You can change the rate of recording frames as you like
            frames.append(env.render())

        next_state, reward, done, _ = env.step(action) 
        episode_reward += reward

        if done:    
            if next_state[0] >= 0.5:
                success = True
            final_position = next_state[0]
            return final_position,success, frames, episode_reward
        else:
            state = next_state


# ### Main Imitation Learning Method (TODO)
# 
# 

# In[ ]:


def imitate(args):
    '''
  TODO: Implement this function

  Input: args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access 

  Functionality: For a given set of args, performs imitation learning as desired. 

  Returns: final_positions, A list of final positions achieved by the model during every time it is tested. Should have a list length of args.max_dagger_iterations
           success_history, A list of success percentage achieved by the model during every time it is tested. Should have a list length of args.max_dagger_iterations
           reward_history, A list of episode rewards achieved by the model during every time it is tested. Should have a list length of args.max_dagger_iterations
           frames, A list of video frames of the model executing its policy every time it is tested, can choose to not record. Should have a length of the number of times you chose to record frames
           args, an object with set of parameters and objects that you can treat as an attribute dictionary. Access elements with args.element_you_want_to_access
  '''

  return final_positions, success_history, frames, reward_history, args


# ### Launch Imitation Learning (TODO)
# Define Args and Launch 'imitate'
# 
# 

# In[ ]:


'''
TODO: Expand the attributes of Args as you please. 

But please maintain the ones given below, i.e: You should be using the ones given below. Fill them out.

Some of these are already filled out for you. 
'''


##TODO: Fill in the given attributes (you should use these in your code), and add to them as you please.
class Args(object):
  pass

args = Args();
args.datapath = 
args.env = 
args.do_dagger = 
args.max_dagger_iterations = 
if not args.do_dagger:
  assert args.max_dagger_iterations==1
args.record_frames = 
args.initial_episodes_to_use = 
args.model = StatesNetwork(args.env)
args.num_epochs = 
args.Q = np.load('./expert_Q.npy',allow_pickle=True)
args.discretization = np.array([10,100])


positions, successes, frames, reward_history, args = imitate(args)


# ### Average Performance Metrics
# 
# Use this function to see how well your agent is doing. 

# In[ ]:


def get_average_performance(args, run_for=1000):

  final_positions = 0
  successes = 0
  rewards = 0

  for ep in range(run_for):
    pos, success, _, episode_rewards = test_model(args, record_frames=False)   #test imitation policy
    final_positions += pos 
    rewards += episode_rewards
    if success:
      successes += 1
    print('Running Episode: ',ep,' Success: ', success)
    average_final_positions = final_positions/(ep+1)
    average_success_rate = 100*(successes/(ep+1))
    average_episode_rewards = rewards/(ep+1)

  return average_final_positions, average_success_rate, average_episode_rewards 


final_pos, succ_rate, ep_rwds = get_average_performance(args)

print('Average Final Position achieved by the Agent: ',final_pos)
print('Average Success Rate achieved by the Agent: ',succ_rate)
print('Average Episode Reward achieved by the Agent: ',ep_rwds)


# ### Visualization

# #### Plotting code
# 
# Use the code below to make plots to see how well your agent did as it trained. 

# In[ ]:


import pandas as pd 

plt.plot(successes)
plt.xlabel('Episode')
plt.ylabel('% of Episodes with Success')
plt.title('% Successes')
plt.show()
plt.close()

p = pd.Series(positions)
ma = p.rolling(3).mean()
plt.plot(p, alpha=0.8)
plt.plot(ma)
plt.xlabel('Episode')
plt.ylabel('Position')
plt.title('Car Final Position')
plt.show()

plt.plot(reward_history)
plt.xlabel('Episode')
plt.ylabel('Episode Rewards Achieved')
plt.title('Episode Rewards')
plt.show()
plt.close()


# #### Make a video!
# 
# Using the frames that you recorded in ``` frames ```, Run the code below to display a video that you can use to see how well your agent is doing
# 
# 

# In[ ]:


#### Video plotting code #####################
deep_frames = []
for f in frames:
  deep_frames += f
plt.figure(figsize=(deep_frames[0].shape[1] / 72.0, deep_frames[0].shape[0] / 72.0), dpi = 72)
patch = plt.imshow(deep_frames[0])
plt.axis('off')
animate = lambda i: patch.set_data(deep_frames[i])
ani = matplotlib.animation.FuncAnimation(plt.gcf(), animate, frames=len(deep_frames), interval = 50)
HTML(ani.to_jshtml())

