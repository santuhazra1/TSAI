## Step: 1
### First we are going to initialized the memory in Replay Buffer class with a max size of 100000 say. Now, we are going to add transitions tuples to it and increment memory pointer 100000 time to fill it for model training to start. We are going to add new transition to 1st place again once all 100000 memory space is full.
![Step 1](Images/Step 1.png)

## Step: 2
### Next we are going to Initialize Actor. There will be two actor as Actor Model and Actor Target. Actor Model is going to get train through backprop and Actor Target is going to get updated once in every two Actor Model update using polyak averaging. Actor Model is going to get updated using backprop once in every two Critic Model update. Initially we will copy all the weights of Actor Model to Actor Target.

## Step: 3
### Now we are going to initialize Critic. There are four sets of Critic as Critic 1 model, Critic 2 model, Critic 1 target and Critic 2 target. Critics Models are going to get updated using back prop and Critic Targets are going to get updated using polyak averaging once in every two actor model update. Initially we will copy all the weights of Critics Model to Critic Targets.

## Step 4:
### Next we are going to randomly sample a batch size of memory touples from replay buffer for training.

##Step 5:
### As we have sampled Memory of length of batch size, now we are going to pass each next state to Actor target to predict next action of each next state.

## Step 6:
### Now we are going to add a bit of gaussian noise to next action to make it more robust to the environment and clamp it to a range supported by the environment.

## Step 7:
### Next we will feed the predicted next action from action target and next state from replay memory batch to Critic 1 Target and Critic 2 Target. Critic 1 Target and Critic 2 Target are going to predict Qt1 and Qt2 values.

## Step 8:
### Now we are going to take the min of Qt1 and Qt2 for the next step.

## Step 9:
### Now we we are going to calculate Qt from min of Qt1 and Qt2 using bellman Eq as:
### Qt = reward + gamma * min(Qt1, Qt2)

## Step 10:
### Next we are going to feed state and action from replay memory batch to Critic 1 Model and Critic 2 Model which are going to predict current Q1 and current Q2.

## Step 11:
### Now we are going to compute critic loss using Q target predicted by Critic target and Current Q's predicted by Critic Model's as:
### Critic Loss = MSEloss(Q1, Qt) + MSEloss(Q2, Qt)

## Step 12:
### Next we are going to back propagate and update the weights of two Critic Models using Critic Loss.

## Step 13:
### Now we are going to update Actor Model by backpropagating actor loss which is computed from first critic Model's output. We are going to update Actor model once in every two iteration.

## Step 14:
### Finally will update actor target and critic target once in every two actor model update using polyak averaging as:

### Actor Target weights = tao x Actor Model weights + (1 - tao) x Actor Target weights

### Critic Target weights = tao x Critic Model weights + (1 - tao) x Critic Target weights

