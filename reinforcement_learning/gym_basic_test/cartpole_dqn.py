from collections import namedtuple
import random
from torch import nn
import torch
import torch.nn.functional as F
from torch import optim
import gym
import matplotlib.pyplot as plt
from matplotlib import animation
import os

# s_t, a_t => s_{t+1}
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.index = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            # placeholder
            self.memory.append(None)
        self.memory[self.index] = Transition(state, action, next_state, reward)
        self.index = (self.index + 1) % self.capacity

    # list of transition
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Q_function base NN
class DQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        # x.shape: batch_size*n_states
        # output.shape: batch_size*n_actions, state_action_value
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class Agent:
    def __init__(
        self, n_states, n_actions, eta=0.5, gamma=0.99, capacity=10000, batch_size=32
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.eta = eta
        self.gamma = gamma
        self.batch_size = batch_size

        self.memory = ReplayMemory(capacity)
        self.model = DQN(n_states, n_actions)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def _replay(self):
        if len(self.memory) < self.batch_size:
            return
        # list of transition
        batch = self.memory.sample(self.batch_size)
        # Transition, column: len(tuple) == batch_size
        batch = Transition(*zip(*batch))

        # s_t.shape: batch_size * 4
        state_batch = torch.cat(batch.state)
        # a_t.shape: batch_size * 1
        action_batch = torch.cat(batch.action)
        # r_{t+1}.shape: batch_size * 1
        reward_batch = torch.cat(batch.reward)
        # < batch_size
        non_final_next_state_batch = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # 构造模型训练用的输入和输出（true）
        # s_t, input

        # pred: Q(s_t, a_t)
        # true: R_{t+1} + \gamma*\max_aQ(s_t, a)

        # 开启 eval 模式
        self.model.eval()

        # pred, batch_size*1
        state_action_values = self.model(state_batch).gather(dim=1, index=action_batch)

        # true: R_{t+1} + \gamma*\max_aQ(s_t, a)
        # tuple(map(lambda s: s is not None, batch.next_state)): batch_size 长度的 0/1
        non_final_mask = torch.ByteTensor(
            tuple(map(lambda s: s is not None, batch.next_state))
        )
        next_state_values = torch.zeros(self.batch_size)
        # Q(s_{t+1}, a)
        next_state_values[non_final_mask] = (
            self.model(non_final_next_state_batch).max(dim=1)[0].detach()
        )

        # (batch_size, )
        expected_state_action_values = reward_batch + self.gamma * next_state_values

        # 开启train mode
        self.model.train()

        # expected_state_action_values.unsqueeze(1): (batch_size, ) => (batch_size, 1)
        loss = F.smooth_l1_loss(
            state_action_values, expected_state_action_values.unsqueeze(1)
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_q_function(self):
        self._replay()

    def memorize(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    # action policy
    # epsilon_greedy
    # double e: explore, exploit
    def choose_action(self, state, episode):
        eps = 0.5 * 1 / (1 + episode)
        if random.random() < eps:
            # explore
            action = torch.IntTensor([[random.randrange(self.n_actions)]])
        else:
            self.model.eval()
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1, 1)
        return action


def display_frames_to_video(frames):
    plt.figure(figsize=(frames[0].shape[0] / 72, frames[0].shape[1] / 72), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate_func(frame):
        patch.set_data(frame)

    anim = animation.FuncAnimation(
        plt.gcf(), animate_func, frames=frames[1:], interval=50
    )
    gif_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gif/cartpole_dqn.gif"
    )
    anim.save(gif_path, writer="imagemagick")


def test():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    max_episodes = 500
    max_steps = 200

    complete_episodes = 0
    finished_flag = False

    agent = Agent(n_states, n_actions)
    frames = []

    for episode in range(max_episodes):
        state = env.reset()[0]
        state = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)
        for step in range(max_steps):
            if finished_flag:
                frames.append(env.render())
            # IntTensor of 1*1
            action = agent.choose_action(state, episode)

            # transition on env
            next_state, _, done, _, _ = env.step(action.item())

            if done:
                next_state = None

                if step < 180:
                    # 1d
                    reward = torch.FloatTensor([-1.0])
                    complete_episodes = 0
                else:
                    reward = torch.FloatTensor([1.0])
                    complete_episodes += 1
            else:
                reward = torch.FloatTensor([0])
                # (4, )
                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                # (4, ) ==> (1, 4)，便于后续的 torch.cat => (1, 4) => (32, 4)
                next_state = next_state.unsqueeze(0)
            agent.memorize(state, action, next_state, reward)
            agent.update_q_function()
            state = next_state

            if done:
                print(f"episode: {episode}, steps: {step}")
                break
        if finished_flag:
            break

        if complete_episodes >= 5:
            finished_flag = True
            print("连续成功10轮")

    # output
    print("frames size:{}".format(len(frames)))
    if len(frames) != 0:
        display_frames_to_video(frames)
    env.close()


if __name__ == "__main__":
    test()
