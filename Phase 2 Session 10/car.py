# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint, choice
import matplotlib.pyplot as plt
import time
import cv2
import math
import torchvision
import torch
import os
import random
from collections import deque

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from kivy.graphics.texture import Texture

# Import TD3
from ai import ReplayBuffer, TD3


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
# action2rotation = [0,5,-5]


im = CoreImage("./images/MASK1.png")

# textureMask = CoreImage(source="./kivytest/simplemask1.png")

# img = PILImage.open("./images/mask.png").convert('L')
# img.show()

# def getpoints(center_x,center_y,angle):

#     a = [center_x - 10, center_y - 5]
#     b = [center_x - 10, center_y + 5]
#     c = [center_x + 10, center_y - 5]
#     d = [center_x + 10, center_y + 5]

#     p, q = center_x, center_y

#     a1 = (int(((a[0] - p) * math.cos(math.radians(angle))) - ((a[1] - q) * math.sin(math.radians(angle))) + p), int(((a[0] - p) * math.sin(math.radians(angle))) + ((a[1] - q) * math.cos(math.radians(angle))) + q ))
#     b1 = (int(((b[0] - p) * math.cos(math.radians(angle))) - ((b[1] - q) * math.sin(math.radians(angle))) + p), int(((b[0] - p) * math.sin(math.radians(angle))) + ((b[1] - q) * math.cos(math.radians(angle))) + q ))
#     c1 = (int(((c[0] - p) * math.cos(math.radians(angle))) - ((c[1] - q) * math.sin(math.radians(angle))) + p), int(((c[0] - p) * math.sin(math.radians(angle))) + ((c[1] - q) * math.cos(math.radians(angle))) + q ))
#     d1 = (int(((d[0] - p) * math.cos(math.radians(angle))) - ((d[1] - q) * math.sin(math.radians(angle))) + p), int(((d[0] - p) * math.sin(math.radians(angle))) + ((d[1] - q) * math.cos(math.radians(angle))) + q ))
#     c = (int((c1[0] + d1[0])/2),int((c1[1] + d1[1])/2))
#     return a1, b1, c


def crop(img, x, y , angle = 0, crop_size = 100, scale_size = 32):
    img = np.asarray(img)
    def pad_with(vector, pad_width, iaxis, kwargs):
            pad_value = kwargs.get('padder', 10)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value    
    img = np.pad(img, crop_size // 2, pad_with, padder=255.)
    img_x, img_y = img.shape
    
    x += crop_size // 2
    y += crop_size // 2
    center_x = x + 10
    center_y = y + 5    

    # pt1,pt2,pt3 = getpoints(center_x,center_y,angle)

    # vertices = np.array([pt1, pt2, pt3], np.int32)
    # pts = vertices.reshape((-1, 1, 2))
    # img = cv2.polylines(img, [pts], isClosed=True, color=(128), thickness=2)

    # img = cv2.fillPoly(img, [pts], color=(128))

    cropped_image = img[center_x - crop_size//2:center_x + crop_size//2,center_y - crop_size//2:center_y + crop_size//2]

    res = cv2.resize(cropped_image, dsize=(scale_size,scale_size), interpolation=cv2.INTER_CUBIC)
    res = np.expand_dims(res, axis=0)
    res = torch.from_numpy(res)
    return res

# crop_img, ori_img = crop(img, 257, 662, 0)
# crop_img = PILImage.fromarray(crop_img)
# ori_img = PILImage.fromarray(ori_img)
# crop_img.show()
# crop_img.show()
# Initializing the map
first_update = True

# Initializing the map
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
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
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal, _ = SelectRandomLocationTarget()
    goal_x, goal_y = goal[0], goal[1]
    first_update = False
    env_name = "SelfDrivingCar-v0" # Name of a environment (set it to any Continous environment you want)
    seed = 0
    start_timesteps = 1e3 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    eval_freq = 5e2 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 4e4 # Total number of iterations/timesteps
    save_models = True # Boolean checker whether or not to save the pre-trained model
    expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 32 # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated



def SelectRandomLocationTarget():
    # Define multiple Target in the dictionary. Total 12 Target defined
    TargetDict = {'A': [115, 215], 'B': [170, 547], 'C': [500, 50], 'D': [743,607], 'E': [650, 310], 'F': [800, 102], 'G': [697, 546], 'H': [1103, 351],\
        'I': [1177, 390], 'J': [1197, 150], 'K': [1326, 27], 'L': [1375, 247]}

    # Define multiple Location in the dictionary. Total 12 Location defined    
    LocationDict =  {'A': [127,297], 'B': [246,105], 'C': [658,248], 'D': [638,77], 'E': [731,386], 'F': [275,608], 'G': [595, 454], 'H': [864,171],\
        'I': [1013,32], 'J': [1297, 304], 'K': [900,88], 'L': [37,571]}

    TargetLocationName = ['A','B','C','D','E','F','G','H','I','J','K','L']

    # Selecting random target and Location
    Target = choice(TargetLocationName)
    Location = choice(TargetLocationName)

    return TargetDict[Target], LocationDict[Location]


# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        reward = 0
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        next_state = crop(sand, self.car.x, self.car.y, self.car.angle)
        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            # print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))            
            reward += -5

        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            reward += -0.5
            # print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
        
        if distance < last_distance and distance!=0:
            reward += 2
        elif distance == 0:
            reward += 100
            done = True
        else:
            reward += -0.2

        if self.car.x < 5:
            self.car.x = 5
            reward += -50
            done = True
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            reward += -50
            done = True
        if self.car.y < 5:
            self.car.y = 5
            reward += -50
            done = True
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            reward += -50 
            done = True     

        return next_state, reward, done


# Creating the game class
# action2rotation = [0,5,-5]

class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        _, self.car.pos = SelectRandomLocationTarget()
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global longueur
        global largeur       

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
        def evaluate_policy(policy, eval_episodes=10):
            avg_reward = 0.
            for _ in range(eval_episodes):
                obs = reset(self)
                done = False
                while not done:
                    action = policy.select_action(obs)
                    obs, reward, done, _ = Car.move(action)
                    avg_reward += reward
            avg_reward /= eval_episodes
            print ("---------------------------------------")
            print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
            print ("---------------------------------------")
            return avg_reward        
        file_name = "%s_%s_%s" % ("TD3", env_name, str(seed))
        print ("---------------------------------------")
        print ("Settings: %s" % (file_name))
        print ("--------------------------------------  -")

        if not os.path.exists("./results"):
            os.makedirs("./results")
        if save_models and not os.path.exists("./pytorch_models"):
            os.makedirs("./pytorch_models")
        
        torch.manual_seed(seed)
        np.random.seed(seed)

        state_dim = [32,32,1]
        action_dim = 1
        max_action = 5

        policy = TD3(state_dim, action_dim, max_action)

        replay_buffer = ReplayBuffer()

        evaluations = [evaluate_policy(policy)]

        def mkdir(base, name):
            path = os.path.join(base, name)
            if not os.path.exists(path):
                os.makedirs(path)
            return path
        work_dir = mkdir('exp', 'brs')
        monitor_dir = mkdir(work_dir, 'monitor')
        max_episode_steps = 400  

        total_timesteps = 0
        timesteps_since_eval = 0
        episode_num = 0
        done = True
        t0 = time.time()  

# We start the main loop over 40,000 timesteps
        while total_timesteps < max_timesteps:
        
            # If the episode is done
            if done:

                # If we are not at the very beginning, we start the training process of the model
                if (total_timesteps != 0 and total_timesteps > (batch_size)):
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
                    policy.train(replay_buffer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)

                # We evaluate the episode and we save the policy
                if timesteps_since_eval >= eval_freq:
                    timesteps_since_eval %= eval_freq
                    evaluations.append(evaluate_policy(policy))
                    policy.save(file_name, directory="./pytorch_models")
                    np.save("./results/%s" % (file_name), evaluations)
                
                # When the training step is done, we reset the state of the environment
                obs = reset()
                
                # Set the Done to False
                done = False
                
                # Set rewards and episode timesteps to zero
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1
        
            # Before 10000 timesteps, we play random actions
            if total_timesteps < start_timesteps:
                action = np.random.normal(0, 1, size=1).clip(-1, 1).astype(np.float32)
            else: # After 10000 timesteps, we switch to the model
                action = policy.select_action(obs)
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if expl_noise != 0:
                    action = (action + np.random.normal(0, expl_noise, size=1)).clip(-1, 1)
            
            # The agent performs the action in the environment, then reaches the next state and receives the reward
            new_obs, reward, done, _ = move(action)
            
            # We check if the episode is done
            # done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            if episode_timesteps + 1 == max_episode_steps:
                done = True
            done = float(done)
            # We increase the total reward
            episode_reward += reward
            
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            obs = new_obs
            episode_timesteps += 1
            total_timesteps += 1 
            timesteps_since_eval += 1

        t1 = time.time()
        print("Total time  taken: {}".format(t1-t0)) 
        evaluations.append(evaluate_policy(policy))
        if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
        np.save("./results/%s" % (file_name), evaluations)        
        CarApp().stop()         

        # rotation = choice(action2rotation)
        # self.car.move(rotation)
        # if self.car.x > self.width - 5:
        #     reset(self)

      

def reset(self):
        _, self.car.pos = SelectRandomLocationTarget()
        self.car.velocity = Vector(6, 0)
        self.car.angle = 0
        state = crop(sand, self.car.x, self.car.y, self.car.angle)
        return state


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
        # brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        # brain.load()

# # # Running the whole thing
if __name__ == '__main__':
    CarApp().run()



