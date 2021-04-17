### Deep Deterministic Policy Gradient

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

The action space is continuous and it is included a vector that has four number between -1 and 1

![image](https://user-images.githubusercontent.com/31414852/115116764-6c29e700-9f69-11eb-82a4-0f89c659bedb.png)

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy.

DDPG trains a deterministic policy in an off-policy method, and due to the policy is deterministic, if the agent were to explore on-policy, in the beginning it would probably not try a wide enough variety of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their actions at training time. 

### Actor Neural Network Architecture

The actor network mapping state to action

- Input Layer: 33
- Hidden Layer 1: 128
- Hidden Layer 2: 128
- Output Layer: 4

```python
Actor(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=33, out_features=128, bias=True)
    (1): Linear(in_features=128, out_features=128, bias=True)
  )
  (output): Linear(in_features=128, out_features=4, bias=True)
)
```


### Critic Neural Network Architecture

The critoc network mapping (state, action) pair to Q-value

- Input Layer: 33
- Hidden Layer 1: 128 + 4
- Hidden Layer 2: 128
- Output Layer: 1

~~~python
Critic(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=33, out_features=128, bias=True)
    (1): Linear(in_features=132, out_features=128, bias=True)
  )
  (output): Linear(in_features=128, out_features=1, bias=True)
)
~~~

The code:

model.py : included the Actor and the Critic classes.They used for the training.

ddpg_agent.py : the ddpg_agent and Replay Buffer memory used.

The learn(): updated the policy and value parameters given batch of experience.

### Hyper-parameters
The result comes from many try and error, it takes so long time to get good performance. I adjusted parameters like network size,learning rate. At the end I choosed the best one to give good performance. During the processing , I did increaseing the number of steps per episode that hepled changing the agent learning, the more the better.It also helps in our case if we put batch normalization in neural network. The last thing is learning rate, a little bit higher values of learning rate which make the agent learn better and easy to solve the problem. Both Neural Networks use the Adam optimizer with a learning rate of 5e-4 and batch size of 128.

```
state_size = 33         # environment State size 
action_size = 4         # environment Action size 
buffer_size = int(1e5)  # replay buffer size
batch_size = 128        # batch size
gamma = 0.99            # discount factor
tau = 1e-3              # for soft update of target parameters
lr_actor = 5e-4         # learning rate of the actor 
lr_critic = 5e-4        # learning rate of the critic
weight_decay = 0        # L2 weight decay
actor_fc1_units = 128   # Number of units for the layer 1 in the actor model
actor_fc1_units = 128   # Number of units for the layer 2 in the actor model
critic_fcs1_units = 128 # Number of units for the layer 1 in the critic model
critic_fc2_units = 128  # Number of units for the layer 2 in the critic model 

Noise:      
mu = 0.                 
theta = 0.15           
sigma = 0.1             
```

### Result

![image](https://user-images.githubusercontent.com/31414852/115101513-d82d3080-9f12-11eb-8cec-8e046cd09ab8.png)

![image](https://user-images.githubusercontent.com/31414852/115101511-cfd4f580-9f12-11eb-993b-4a697c1fae1e.png)

ddpg method solved the problem in 591 episodes.


## Future Improvement

- Tuning hyper parameters to get better performance
- Make the network more layers to  get good performance
- Continue working with Policy Gradient algorithm like A3C, A2C, PPO
