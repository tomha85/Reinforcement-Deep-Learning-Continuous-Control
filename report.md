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
- Hidden Layer 1: 256 + 4
- Hidden Layer 2: 128
- Output Layer: 1

~~~python
Critic(
  (hidden_layers): ModuleList(
    (0): Linear(in_features=33, out_features=256, bias=True)
    (1): Linear(in_features=260, out_features=128, bias=True)
  )
  (output): Linear(in_features=128, out_features=1, bias=True)
)
~~~



### Hyper-parameters

- Replay Memory Size = 1e5
- Batch Size = 128
- GAMMA = 0.99
- TAU = 1e-3
- Actor Learning Rate = 1e-3
- Critic Learning Rate = 1e-4
- Noise Decaying Rate = 0.99

### Result

![DDPG Scores][image1]

DDPG solved the problem in 199 episodes.


## Future Improvement

- Fine tuning hyper parameters to get better performance;
- Try to make the network deeper to  get better performance;
- Try other Policy Gradient Algorithms, such as PPO, A3C
