import gym
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os


# CartPole-v1官方示例，动作空间离散，状态空间连续
class Agent:
    def __init__(self, action_space, n_states, eta=0.5, gamma=0.99, NUM_DIGITIZED=6):
        self.eta = 0.5
        self.gamma = gamma
        # Discrete(2)
        self.action_space = action_space
        self.NUM_DIGITIZED = NUM_DIGITIZED
        self.q_table = np.random.uniform(
            0, 1, size=(NUM_DIGITIZED**n_states, self.action_space.n)
        )

    # 分桶， 5个值，对应 6 个分段，即 6 个桶 (0, 1, 2, 3, 4, 5)
    @staticmethod
    def _bins(clip_min, clip_max, num_bins):
        return np.linspace(clip_min, clip_max, num_bins + 1)[1:-1]

    # 按 6 进制映射将 4位 6 进制数映射为 id，
    @staticmethod
    def _digitize_state(observation, NUM_DIGITIZED):
        pos, cart_v, angle, pole_v = observation
        # print("check {},{},{},{}".format(pos, cart_v, angle, pole_v))
        digitized = [
            np.digitize(pos, bins=Agent._bins(-2.4, 2.4, NUM_DIGITIZED)),
            np.digitize(cart_v, bins=Agent._bins(-3.0, 3, NUM_DIGITIZED)),
            np.digitize(angle, bins=Agent._bins(-0.418, 0.418, NUM_DIGITIZED)),
            np.digitize(pole_v, bins=Agent._bins(-2, 2, NUM_DIGITIZED)),
        ]
        # 3,1,2,4 (4位10进制数) = 4*10^0 + 2*10^1 + 1*10^2 + 3*10^3，最终的取值范围是 0-9999，总计 10^4 == 10000
        # a,b,c,d (4位6进制数) = d*6^0 + c*6^1 + b*6^2 + a*6^3，最终的取值范围是 0-`5555`(1295)，总计 6^4 == 1296
        ind = sum([d * (NUM_DIGITIZED**i) for i, d in enumerate(digitized)])
        return ind

    def q_learning(self, obs, action, reward, obs_next):
        obs_ind = self._digitize_state(obs, self.NUM_DIGITIZED)
        obs_next_ind = self._digitize_state(obs_next, self.NUM_DIGITIZED)
        self.q_table[obs_ind, action] = self.q_table[obs_ind, action] + self.eta * (
            reward + max(self.q_table[obs_next_ind, :]) - self.q_table[obs_ind, action]
        )

    def choose_action(self, state, episode):
        # 根据q表选择动作或者在动作空间随机采样
        eps = 0.5 * 1 / (episode + 1)  # 随着episode增加而降低
        state_ind = Agent._digitize_state(state, self.NUM_DIGITIZED)
        # epsilon greedy
        if np.random.random() < eps:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.q_table[state_ind, :])
        return action


def test():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()
    action_space = env.action_space
    n_states = env.observation_space.shape[0]
    agent = Agent(action_space, n_states)

    max_episodes = 1000
    success_episodes_thresh = 10
    max_steps = 300
    design_steps_thresh = 295
    continue_success_episodes = 0
    learning_finish_flag = False
    frames = []
    for episode in range(max_episodes):
        obs = env.reset()[0]
        for step in range(max_steps):
            if learning_finish_flag:
                frames.append(env.render())
            # 自定义策略PI
            action = agent.choose_action(obs, episode)
            # 内置step，根据动作转移到下一个状态
            obs_next, _, done, _, _ = env.step(action)
            # 不采用内置step返回的reward，自定义reward
            if done:
                # design_steps_thresh之内死掉了，奖励为负
                if step < design_steps_thresh:
                    reward = -1
                    continue_success_episodes = 0
                else:
                    reward = 1
                    continue_success_episodes += 1
                    print(
                        "-> continue_success_episodes:{}".format(
                            continue_success_episodes
                        )
                    )
            else:
                if step == max_steps - 1:
                    reward = 1
                    continue_success_episodes += 1
                    print(
                        "-> continue_success_episodes:{}".format(
                            continue_success_episodes
                        )
                    )
                else:
                    reward = 0
            # 训练一次q表
            agent.q_learning(obs, action, reward, obs_next)
            # 该step结束
            if done:
                print(f"episode: {episode}, finish {step} time steps.(break)")
                break
            if step == max_steps - 1:
                print(f"episode: {episode}, finish {step} time steps.(done)")
            # 迭代更新状态
            obs = obs_next

        if learning_finish_flag:
            break
        if continue_success_episodes >= success_episodes_thresh:
            learning_finish_flag = True
            print(
                "continue success more than {} times ".format(success_episodes_thresh)
            )

    # output
    print("frames size:{}".format(len(frames)))
    if len(frames) != 0:
        display_frames_to_video(frames)
    env.close()


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
        os.path.dirname(os.path.abspath(__file__)), "cartpole_q_learning.gif"
    )
    anim.save(gif_path, writer="imagemagick")


if __name__ == "__main__":
    test()

    print("the end")
