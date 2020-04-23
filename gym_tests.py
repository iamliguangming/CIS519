import gym
import numpy as np
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
  total_Reward = 0
  frame = []
  for i in range(num_episodes):
    env.reset()
    for j in range(200):
      env.render()
      print(env.render())
    #   current_Frame = env.render(mode = 'rgb_array')
      if i == num_episodes - 1:
        frame.append(env.render(mode = 'rgb_array'))
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      if done or j==199:
        total_Reward += -j
        break
  env.close()
  mean_reward = total_Reward / num_episodes




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