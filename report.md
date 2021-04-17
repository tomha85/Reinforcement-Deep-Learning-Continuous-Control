### Deep Deterministic Policy Gradient

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

The action space is continuous and it is included a vector that has four number between -1 and 1


### Actor Neural Network Architecture

The actor network mapping state to action

- Input Layer: 33
- Hidden Layer 1: 256
- Hidden Layer 2: 128
- Output Layer: 4

```python
Actor(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=33, out_features=128, bias=True)
    (1): Linear(in_features=512, out_features=128, bias=True)
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



### Hyper-parameters

state_size = 33         # environment State size 
action_size = 4         # environment Action size 
buffer_size = int(1e5)  # replay buffer size
batch_size = 128        # minibatch size

gamma = 0.99            # discount factor
tau = 1e-3              # for soft update of target parameters

lr_actor = 5e-4         # learning rate of the actor 
lr_critic = 5e-4        # learning rate of the critic
weight_decay = 0        # L2 weight decay
actor_fc1_units = 128   # Number of units for the layer 1 in the actor model
actor_fc1_units = 128   # Number of units for the layer 2 in the actor model
critic_fcs1_units = 128 # Number of units for the layer 1 in the critic model
critic_fc2_units = 128  # Number of units for the layer 2 in the critic model
bn_mode = 2             # Use Batch Norm. - 0=disabled, 1=BN before Activation, 2=BN after Activation 
                        # (3, 4 are alt. versions of 1, 2)

add_ounoise = True      # Add Ornstein-Uhlenbeck noise
mu = 0.                 # Ornstein-Uhlenbeck noise parameter
theta = 0.15            # Ornstein-Uhlenbeck noise parameter
sigma = 0.1             # Ornstein-Uhlenbeck noise parameter

### Result

![DDPG Scores][image1]

DDPG solved the problem in 199 episodes.


## Future Improvement

- Fine tuning hyper parameters to get better performance;
- Try to make the network deeper to  get better performance;
- Try other Policy Gradient Algorithms, such as PPO, A3C
