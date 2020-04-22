# Project 1 Report

## Learning Algorithm

The implementation is a Dueling Deep Q-Network with three fully connected layers: two hidden layers of 128 units each with ReLU activation and an output layer of 4 units (one for each action). The hyperparameters are as follows:

| Hyperparameter | Value |
| ------------- | ------------- |
| first hidden layer units | 128 |
| second hidden layer units | 128 |
| replay buffer size | 1e6 |
| batch size | 64 |
| discount factor (gamma) | 0.99 |
| tau* | 1e-3 |
| learning rate | 25e-5 |
| update the network every __ time steps | 4 |
| number of episodes | 1000 |
| max time steps per episode | 1000 |
| starting epsilon value | 1.0 |
| epsilon decay rate | 0.95 |

*Tau is the percentage of weights from the local model to carry over to the target model during the soft update of target parameters; meanwhile, `1 - tau` is the percentage of target model weights to carry over.

## Plot of Rewards

The plot below shows that, after 255 episodes, the agent is able to receive an average reward of 13 over the last 100 consecutive episodes.

![final_model_rewards_plot](./download(1).png)

We optimized our code with few trials of hyperparameter tuning and got good results with epsilon decay of 0.99 and got a result of around 255 episodes. Our final weights are [this](./model_weights1.pth)

## Ideas for Future Work

Ideas for improving the agent's performance are as follows:
- Solve the environment in fewer than 200 episodes.
- Applying priortized replay, Double DQN and Rainbow DQN.
