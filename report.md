In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.


### Result

![DDPG Scores][image1]

DDPG solved the problem in 199 episodes.


## Future Improvement

- Fine tuning hyper parameters to get better performance;
- Try to make the network deeper to  get better performance;
- Try other Policy Gradient Algorithms, such as PPO, A3C
