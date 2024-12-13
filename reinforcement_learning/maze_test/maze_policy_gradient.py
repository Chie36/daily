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
        if action == 0:
            self.state -= 3
        elif action == 1:
            self.state += 1
        elif action == 2:
            self.state += 3
        elif action == 3:
            self.state -= 1
        return self.state, 1, self.state == 8, False, {}


# 动作策略选择，基于当前环境的状态
class Agent:
    def __init__(self):
        self.actions = list(range(4))
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
        self.theta = self.theta_0
        self.pi = self._softmax_cvt_theta_to_pi()
        # self.pi = self._cvt_theta_to_pi()

        self.eta = 0.1

    def _cvt_theta_to_pi(self):
        # theta → pi
        m, n = self.theta.shape
        pi = np.zeros((m, n))
        for r in range(m):
            pi[r, :] = self.theta[r, :] / np.nansum(self.theta[r, :])
        return np.nan_to_num(pi)

    def _softmax_cvt_theta_to_pi(self, beta=2.0):
        # theta → pi
        m, n = self.theta.shape
        pi = np.zeros((m, n))
        exp_theta = np.exp(self.theta * beta)
        for r in range(m):
            pi[r, :] = exp_theta[r, :] / np.nansum(exp_theta[r, :])
        return np.nan_to_num(pi)

    def update_theta(self, s_a_history):
        # theta → theta
        T = len(s_a_history) - 1
        m, n = self.theta.shape
        delta_theta = self.theta.copy()
        for i in range(m):
            for j in range(n):
                if not (np.isnan(self.theta_0[i, j])):
                    sa_i = [sa for sa in s_a_history if sa[0] == i]
                    sa_ij = [sa for sa in s_a_history if (sa[0] == i and sa[1] == j)]
                    N_i = len(sa_i)
                    N_ij = len(sa_ij)
                    delta_theta[i, j] = (N_ij - self.pi[i, j] * N_i) / T
        self.theta = self.theta + self.eta * delta_theta
        return self.theta

    def update_pi(self):
        self.pi = self._softmax_cvt_theta_to_pi()
        return self.pi

    def choose_action(self, state):
        # 依概率分布 \pi_\theta 选择 action
        action = np.random.choice(self.actions, p=self.pi[state, :])
        return action


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
    # 策略迭代（策略梯度），RL包括policy based和Value based，这里是 policy based
    # pi是策略，theta是策略参数，theta_0 → pi_0, theta_0 → theta_1, theta_1 → pi_1
    stop_eps = 1e-3
    agent = Agent()
    env = MazeEnv()
    cnt = 1
    while True:
        # 不断地从初始状态出发，产生一次 trajectory
        state = env.reset()
        # state, action
        s_a_history = [[state, np.nan]]
        while True:
            action = agent.choose_action(state)
            s_a_history[-1][1] = action
            state, reward, terminated, _, _ = env.step(action)
            s_a_history.append([state, np.nan])
            if terminated:
                break

        # 更新 theta
        agent.update_theta(s_a_history)
        pi = agent.pi.copy()
        # 更新 pi
        agent.update_pi()
        delta = np.sum(np.abs(agent.pi - pi))
        # print(
        #     "delta:{}, s_a_history({}):{}".format(delta, len(s_a_history), s_a_history)
        # )

        if delta < stop_eps:
            gif_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "maze_policy_gradient.gif"
            )
            vis(s_a_history, gif_path)
            print(
                "stop at step:{}\npi:\n{}\ntheta:\n{}".format(
                    cnt, agent.pi, agent.theta
                )
            )
            break
        # vis(s_a_history, "policy_gradient_epoch_" + str(cnt) + ".gif")
        cnt += 1
    print("the end")
