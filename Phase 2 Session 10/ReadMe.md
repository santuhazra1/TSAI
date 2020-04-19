# **Phase 2 Assignment 10**

### In this assignemnt we are going to implement TD3 in assignment 7 car environment. So, here are the required steps that has been fllowed to slove it:

### _**Architecture:**_ I am planning to write kivy environment details inside car.py. I am planning to implement TD3 inside ai.py. Now in car.py inside Game calss I am implementing training process of the car. For each action my move function inside Car class should return next state, reward and done and reset is going to reset my car location to random predefined point every time done is true. Crop function should return scaled croped image with a tilted triangle inplace of a car.

### _**Step 1:**_ Removed all sensors information from TD3 map and kivy environment because now we have to crop a image sorrounding the car by looking at which out TD3 agent should predict action.

### _**Step 2:**_ In assignment 7 code our car was always starting from center of the map. We have created SelectRandomLocationTarget function where we have defined multiple location on image. Now, at a time our model is going to pick a random location at a time for better learning.

### _**Step 3:**_ Changed the ai.py with respect to TD3 so that we can call it from our main car.py code

### _**Step 4:**_ Changed actor and critic to a CNN model with a input image size of 32 x 32.

### _**Step 5:**_ Created a reset function so that if our done parameter is true each time it will initialize the car at a random location with initial velocity and zero angle and it would return initial croped image observation.

### _**Step 6:**_ Created a crop function to crop the sand with respect to location of the car with a size 100 x 100 and resized it to 32 x 32.

### _**Step 7:**_ Defined reward and done parameter inside Car classes's move function so that if we call car.move it should change position of the car with respect to velocity and angle and it should return next obs, reward and done parameter.

### _**Step 8:**_ Defined done parameter as when car hits the wall, when car reaches the target and if car don't reach target after 400 steps.

### _**Step 9:**_  Now to run the taining process I have started while inside update function of Game class and initialized all global variable. Now, for each time stamp first it check done parameter if it is True then it will call train and reset the environment as one episode will be over.

### _**Step 10:**_ For each time stamp it should select action for the current step with the help of actor. Now, with that action we will move function to update car location and it will return next state, reward and done for that step.

## **NOTE:** Till here I have implemented as you can see in my code. Still I am stuck with some error while train. I am working on it. Also, while cropping car image sending a tilted triangle with car's angle part I am working on. Its almost done and commented in my code as you can see. Still some correction is needed for co-ordinates of the triangle. And also I have plan to explore the distribution of rewards with a graph. If I get some extended time I will be able to finish it properly.
