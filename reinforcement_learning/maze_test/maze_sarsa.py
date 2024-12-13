import gym
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os


# 维护着状态，以及 step 函数的返回
class MazeEnv(gym.Env):
    def __init__(self):
        self.state = 0
        pass

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        # 这个其实就是状态转移函数
        if action == 0:
            self.state -= 3
        elif action == 1:
            self.state += 1
        elif action == 2:
            self.state += 3
        elif action == 3:
            self.state -= 1
        is_goal = self.state == 8
        return self.state, 1 if is_goal else 0, is_goal, False, {}


# 动作策略选择，基于当前环境的状态
class Agent:
    def __init__(self):
        self.action_space = list(range(4))
        self.theta_0 = np.asarray(
            [
                [np.nan, 1, 1, np.nan],  # s0
                [np.nan, 1, np.nan, 1],  # s1
                [np.nan, np.nan, 1, 1],  # s2
                [1, np.nan, np.nan, np.nan],  # s3
                [np.nan, 1, 1, np.nan],  # s4
                [1, np.nan, np.nan, 1],  # s5
                [np.nan, 1, np.nan, np.nan],  # s6
                [1, 1, np.nan, 1],
            ]  # s7
        )
        self.pi = self._cvt_theta_to_pi()
        self.Q = np.random.rand(*self.theta_0.shape) * self.theta_0
        self.eta = 0.1
        self.gamma = 0.9
        self.eps = 0.5

    def get_action(self, s):
        # eps, explore
        if np.random.rand() < self.eps:
            # 根据策略pi（既概率密度函数，依概率采样动作，得到概率最大的那个动作）
            action = np.random.choice(self.action_space, p=self.pi[s, :])
        else:
            # 1-eps, exploit
            # 直接从Q表中得到当前状态下Q值最高的那个动作
            action = np.nanargmax(self.Q[s, :])
        return action

    def sarsa(self, s, a, r, s_next, a_next):
        # 用于更新Q表
        if s_next == 8:
            self.Q[s, a] = self.Q[s, a] + self.eta * (r - self.Q[s, a])
        else:
            self.Q[s, a] = self.Q[s, a] + self.eta * (
                r + self.gamma * self.Q[s_next, a_next] - self.Q[s, a]
            )

    def _cvt_theta_to_pi(self):
        # theta → pi
        m, n = self.theta_0.shape
        pi = np.zeros((m, n))
        for r in range(m):
            pi[r, :] = self.theta_0[r, :] / np.nansum(self.theta_0[r, :])
        return np.nan_to_num(pi)


def vis(s_a_history, filename):

    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # plt.plot([1, 1], [0, 1], color='red', linewidth=2)
    # plt.plot([1, 2], [2, 2], color='red', linewidth=2)
    # plt.plot([2, 2], [2, 1], color='red', linewidth=2)
    # plt.plot([2, 3], [1, 1], color='red', linewidth=2)

    plt.plot([2, 3], [1, 1], color="red", linewidth=2)
    plt.plot([0, 1], [1, 1], color="red", linewidth=2)
    plt.plot([1, 1], [1, 2], color="red", linewidth=2)
    plt.plot([1, 2], [2, 2], color="red", linewidth=2)

    plt.text(0.5, 2.5, "S0", size=14, ha="center")
    plt.text(1.5, 2.5, "S1", size=14, ha="center")
    plt.text(2.5, 2.5, "S2", size=14, ha="center")
    plt.text(0.5, 1.5, "S3", size=14, ha="center")
    plt.text(1.5, 1.5, "S4", size=14, ha="center")
    plt.text(2.5, 1.5, "S5", size=14, ha="center")
    plt.text(0.5, 0.5, "S6", size=14, ha="center")
    plt.text(1.5, 0.5, "S7", size=14, ha="center")
    plt.text(2.5, 0.5, "S8", size=14, ha="center")
    plt.text(0.5, 2.3, "START", ha="center")
    plt.text(2.5, 0.3, "GOAL", ha="center")
    # plt.axis('off')
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        right=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    (line,) = ax.plot([0.5], [2.5], marker="o", color="g", markersize=60)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        state = s_a_history[i][0]
        x = (state % 3) + 0.5
        y = 2.5 - int(state / 3)
        line.set_data(x, y)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(s_a_history),
        interval=200,
        repeat=False,
    )
    anim.save(filename, writer="imagemagick")


if __name__ == "__main__":
    agent = Agent()
    env = MazeEnv()
    epoch = 0
    while True:
        old_Q = np.nanmax(agent.Q, axis=1)
        # state, action
        s = env.reset()
        a = agent.get_action(s)
        s_a_history = [[s, np.nan]]
        while True:
            s_a_history[-1][1] = a
            s_next, reward, terminated, _, _ = env.step(a)
            s_a_history.append([s_next, np.nan])
            a_next = np.nan if terminated else agent.get_action(s_next)
            agent.sarsa(s, a, reward, s_next, a_next)
            if terminated:
                break
            else:
                a = a_next
                s = s_next
        # s_s_history, agent.Q
        update = np.sum(np.abs(np.nanmax(agent.Q, axis=1) - old_Q))
        epoch += 1
        agent.eps /= 2
        print(
            "epoch:{}, update:{}, traj_size:{}".format(epoch, update, len(s_a_history))
        )
        if epoch > 100 or update < 1e-2:
            gif_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "maze_sarsa.gif"
            )
            vis(s_a_history, gif_path)

            break

    print("the end")
