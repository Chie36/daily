import torch
from torch import nn
from torch import optim
from torch import autograd
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gym
import os


class PolicyNetwork(nn.Module):
    def __init__(self, num_states, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.fc1 = nn.Linear(num_states, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        # x: state, 批次化的, x.shape == (batch, num_states), batch: 一次完整的 trajectory 经历的 steps
        # num_states of cartpole: https://www.gymlibrary.dev/environments/classic_control/cart_pole/
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        # return：(batch, num_actions)， 行和为1；
        return x

    def choose_action(self, state):
        # given state
        # state.shape (4, ), 1d numpy ndarray
        # state, (1, 4)
        state = torch.from_numpy(state).float().unsqueeze(0)
        # probs, (1, 2)
        #         probs = self.forward(autograd.Variable(state))
        probs = self.forward(state)
        # 以概率采样
        highest_prob_action = np.random.choice(
            self.num_actions, p=np.squeeze(probs.detach().numpy())
        )
        # prob => log prob
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        # log_p < 0
        return highest_prob_action, log_prob


GAMMA = 0.9


# rewards 由一次 episode 的 trajectory 产生
def discounted_future_reward(rewards):
    discounted_rewards = []
    for t in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[t:]:
            Gt += (GAMMA**pw) * r
            pw += 1
        discounted_rewards.append(Gt)
    # len(discounted_rewards) == len(rewards)
    return discounted_rewards


def update_policy(policy_network, rewards, log_probs):
    # len(rewards) == len(log_probs)
    # Gt
    discounted_rewards = discounted_future_reward(rewards)

    # normalize discounted rewards => stability
    # from one episode
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
        discounted_rewards.std() + 1e-9
    )

    policy_grads = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_grads.append(-log_prob * Gt)

    policy_network.optimizer.zero_grad()
    policy_grad = torch.stack(policy_grads).sum()
    policy_grad.backward()
    policy_network.optimizer.step()


def test():
    env = gym.make("CartPole-v0")
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)

    max_episodes = 2000
    max_steps = 500

    num_steps = []
    avg_num_steps = []
    all_rewards = []

    for episode in range(max_episodes):
        state = env.reset()[0]
        log_probs = []
        rewards = []
        for step in range(max_steps):
            # $\pi_\theta(a_t|s_t)$
            action, log_prob = policy_net.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                # 完成一次 episode/rollout，得到一次完整的 trajectory
                update_policy(policy_net, rewards, log_probs)
                num_steps.append(step)
                avg_num_steps.append(np.mean(num_steps[-10:]))
                all_rewards.append(sum(rewards))
                if episode % 100 == 0:
                    print(
                        f"episode: {episode}, total reward: {sum(rewards)}, average_reward: {np.mean(all_rewards)}, length: {step}"
                    )
                break
            state = next_state
    plt.plot(num_steps)
    plt.plot(avg_num_steps)
    plt.legend(["num_steps", "avg_steps"])
    plt.xlabel("episode")
    plt.show()
