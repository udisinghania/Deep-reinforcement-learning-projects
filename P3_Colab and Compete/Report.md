# Project 3 Report

## Learning Algorithm

In this project, we used the Deep Deterministic Policy Gradients ([DDPG](https://arxiv.org/abs/1509.02971)) learning algorithm to solve the Tennis environment. DDPG is an off-policy model-free algorithm that uses neural networks to learn policies, even in high-dimensional and continuous action spaces. The implementation has two DDPG agents with shared actor and critic networks. Each agent uses the same actor network to take an action, sampled from a shared replay buffer. Both actor and critic have three fully connected layers: hidden layers of 128 and then 64 units, each with ReLU activation, and an output layer (of 2 units for the actor, with tanh applied in order to bound the output between -1 and 1, and of 1 unit for the critic). In addition, in the critic network, the action vector is included between the first and second hidden layers. The hyperparameters are as follows:

| Hyperparameter | Value |
| ------------- | ------------- |
| replay buffer size | 1e6 |
| batch size | 1024 |
| discount factor (gamma) | 0.99 |
| tau* | 1e-3 |
| actor learning rate | 1e-4 |
| critic learning rate | 1e-3 |
| number of episodes | 3000 |
| L2 weight decay | 0 |

*Tau is the percentage of weights from the local model to carry over to the target model during the soft update of target parameters; meanwhile, `1 - tau` is the percentage of target model weights to carry over.

## Results

| Trial | # of Episodes to Solve | Description | Comments |
| ------------- | ------------- | ------------- | ------------- |
| [Initial Run] | 756 | Default params with tau 1e-3, batch size 1024, and critic LR 1e-3 | Baseline Reacher code with a change to tau |
| [Trial 2] | 789 | Tau 1e-1 | Not better than Initial Run |
| [Trial 3] | 1034 | Tau 3e-1 and critic LR 1e-4 | Better than Initial Run |
| [Trial 4] | 865 | Actor and Critic models without Batch norm, and also adding leakyReLU in critic method AND fc units from 128, 64 to 256 AND also tau= 1e-3 and critic lr = 1e-3 and buffer size = 1e5 | Better than Trial 3 |
| [Trial 5] | 1284 | Going back to original model with Gaussian noise| Worst results thus far |
| [Final Run] | 756 | Default params with tau 1e-3, batch size 1024, and critic LR 1e-3 | Verifying Initial Run performs best |

## Plot of Rewards

The plot below shows that, after 756 episodes, the agent is able to receive an average reward of 0.5 over the last 100 consecutive episodes.

![final_model_rewards_plot](./final_model_rewards_plot.png)

## Ideas for Future Work

Ideas for improving the agent's performance are as follows:
- Attempt prioritized experience replay and D4PG.
- Add lots of noise at the beginning and then reduce or remove it completely after a certain number of episodes.
- Try to solve it within 500 episodes.
- Implement MADDPG: separate actors, separate centralized critics, and a shared replay buffer.
- Try a variation with: separate actors, one shared centralized critic, and a shared replay buffer.
