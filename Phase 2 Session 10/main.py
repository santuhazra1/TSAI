# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint, choice
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import os
import random
from collections import deque
from PIL import Image as PILImage
import torch

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty, BoundedNumericProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture

from TD3 import ReplayBuffer, TD3
from crop import CropImage
from randomLoc import RandomLocation


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0
last_distance = 0

im = CoreImage("./images/MASK1.png")

cropimage = CropImage()
randomlocation = RandomLocation()

if not os.path.exists("./results"):
    os.makedirs("./results")
if not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

env_name = "SelfDrivingCar-v0"
seed = 0
file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
state_dim = [1,32,32]
action_dim = 1
max_action = 5.0

policy = TD3(state_dim, action_dim, max_action)
policy.load(file_name, './pytorch_models/')
max_distance = np.sqrt((1429)**2 + (660)**2)

replay_buffer = ReplayBuffer()

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

# Initializing the map
first_update = True

# Initializing the map
def init():
    global sand
    global first_update
    global last_distance
    global env_name
    global seed
    global start_timesteps
    global eval_freq
    global max_timesteps
    global save_models
    global expl_noise
    global batch_size
    global discount
    global tau
    global policy_noise
    global noise_clip
    global policy_freq
    global done_bool
    global file_name
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    # img = np.zeros([1429, 660],dtype=np.uint8)
    sand = np.asarray(img)/255.0
    first_update = False
    env_name = "SelfDrivingCar-v0" # Name of a environment (set it to any Continous environment you want)
    seed = 0
    start_timesteps = 5000 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 5e2 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 4e4 # Total number of iterations/timesteps
    save_models = True # Boolean checker whether or not to save the pre-trained model
    expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 16 # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated
    file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
    print ("---------------------------------------")
    print ("Settings: %s" % (file_name))
    print ("--------------------------------------  -")

class Goal(Widget):
    pass

# Creating the car class

class Car(Widget):
    
    angle = BoundedNumericProperty(0)
    rotation = BoundedNumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation, episode_timesteps):
        global last_distance

        self.car.pos = Vector(*self.car.velocity) + self.car.pos
        self.car.rotation = rotation
        self.car.angle = self.car.angle + self.car.rotation
        reward = 0
        done = False
        distance = np.sqrt((self.car.x - self.goal.x)**2 + (self.car.y - self.goal.y)**2)
        xx = self.goal.x - self.car.x
        yy = self.goal.y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.        
        next_state = cropimage.crop(sand, self.car.center_x, self.car.center_y, self.car.angle)
        if sand[int(self.car.x - 1),int(self.car.y - 1)] > 0.5:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)          
            reward += -1
        elif sand[int(self.car.x - 1),int(self.car.y - 1)] > 0.1:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle) 
            reward += -0.5
        else: # otherwise
            self.car.velocity = Vector(3, 0).rotate(self.car.angle)
            reward += -0.1
        
        if distance < last_distance:
            reward += 1
        else:
            reward += -0.1            
        
        if distance <= 15:
            reward += 400
            done = True

        if self.car.x < 5:
            self.car.x = 5
            reward += -500
            done = True
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            reward += -500
            done = True
        if self.car.y < 5:
            self.car.y = 5
            reward += -500
            done = True
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            reward += -500
            done = True     

        last_distance = distance 
        if episode_timesteps + 1 == 500:
            done = True
            # reward += -50
        else:
            episode_timesteps += 1

        if done == True: 
            episode_timesteps += 1          
        # print("Episode TimeStep: {} rotation: {} angle:{} reward: {} done: {}".format(episode_timesteps, rotation, self.car.angle, reward, done ))
        distance_norm = distance / max_distance
        return next_state, reward, done, episode_timesteps, distance_norm, orientation

total_timesteps = 0
episode_num = 0
# Creating the game class
class Game(Widget):

    car = ObjectProperty(None)
    goal = ObjectProperty(None)

    def serve_car(self):
        self.car.pos = randomlocation.carlocation()
        self.car.velocity = Vector(6, 0)
        self.goal.pos = randomlocation.target()

    def update(self, dt):

        global longueur
        global largeur   
        global done    
        global episode_reward
        global episode_timesteps
        global episode_num
        global total_timesteps
        global obs
        global dist_curr
        global ori_curr

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            done = True
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        # We start the main loop over 40,000 timesteps
        if total_timesteps < max_timesteps:
        
            # If the episode is done
            if done:
                # If we are not at the very beginning, we start the training process of the model
                if (total_timesteps != 0 and total_timesteps > (batch_size)):
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
                
                # When the training step is done, we reset the state of the environment
                obs, dist_curr, ori_curr = reset(self)
                
                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
        
            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = (5.0 * (random.randint(-100000000, 100000000)/100000000))
            else: # After 10000 timesteps, we switch to the model
                action = np.array(policy.select_action(np.expand_dims(obs, axis=0), np.array(dist_curr), np.array(ori_curr)), dtype=np.float32)
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=1)).clip(-max_action, max_action) 
                action = action[0]
        
            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done, episode_timesteps, dist_nxt, ori_nxt = Car.move(self, action, episode_timesteps)
            
            # We check if the episode is done
            # done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            # if episode_timesteps + 1 == max_episode_steps:
            #     done = True
            done_bool = float(done)
            # We increase the total reward
            episode_reward += reward
            
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward, dist_curr, dist_nxt, ori_curr, ori_nxt, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            dist_curr = dist_nxt
            ori_curr = ori_nxt
            # episode_timesteps += 1
            total_timesteps += 1 

        #     if episode_num == 10:
        #         policy.save("%s" % (file_name), directory="./pytorch_models") 
        # else:
        #     if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")     
        #     CarApp().stop()         


      
def reset(self):  
        self.car.pos = randomlocation.carlocation()
        self.car.velocity = Vector(6, 0)
        self.car.angle = 0
        self.goal.pos = randomlocation.target()  
        distance = np.sqrt((self.car.x - self.goal.x)**2 + (self.car.y - self.goal.y)**2)
        distance_norm = distance / max_distance
        xx = self.goal.x - self.car.x
        yy = self.goal.y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.             
        state = cropimage.crop(sand, self.car.x, self.car.y, self.car.angle)
        return state, distance_norm, orientation


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        policy.save("%s" % (file_name), directory="./pytorch_models")  
        # plt.plot(scores)
        # plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        policy.load("%s" % (file_name), directory="./pytorch_models")  

# # # Running the whole thing
if __name__ == '__main__':
    CarApp().run()



