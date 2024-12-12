import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import os


def map():
    fig = plt.figure(figsize=(5, 5))
    ax = plt.gca()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)

    # plt.plot([1, 1], [0, 1], color="red", linewidth=2)
    # plt.plot([1, 2], [2, 2], color="red", linewidth=2)
    # plt.plot([2, 2], [2, 1], color="red", linewidth=2)
    # plt.plot([2, 3], [1, 1], color="red", linewidth=2)

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
    plt.show()
    return fig, line


def cvt_theta_to_pi(theta):
    # 返回：每一行表示每个状态，每一列表示该状态下每一个动作选择的概率分布
    m, n = theta.shape
    pi = np.zeros((m, n))
    for r in range(m):
        pi[r, :] = theta[r, :] / np.nansum(theta[r, :])
    return np.nan_to_num(pi)


def step(state, action):
    # 对应地图上状态变化，如从s2到s5需要向下移动3
    if action == 0:
        state -= 3
    elif action == 1:
        state += 1
    elif action == 2:
        state += 3
    elif action == 3:
        state -= 1
    return state


if __name__ == "__main__":
    # state & action matrix, action[↑,→,↓,←]
    theta_0 = np.asarray(
        [
            [np.nan, 1, 1, np.nan],  # s0
            [np.nan, 1, np.nan, 1],  # s1
            [np.nan, np.nan, 1, 1],  # s2
            [1, np.nan, np.nan, np.nan],  # s3
            [np.nan, 1, 1, np.nan],  # s4
            [1, np.nan, np.nan, 1],  # s5
            [np.nan, 1, np.nan, np.nan],  # s6
            [1, 1, np.nan, 1],  # s7
        ]
    )

    fig, line = map()
    pi = cvt_theta_to_pi(theta_0)

    state = 0
    state_his = [state]
    action_his = []
    while True:
        action = np.random.choice([0, 1, 2, 3], p=pi[state, :])
        state = step(state, action)
        action_his.append(action)
        state_his.append(state)
        if state == 8:
            break
    print("state_his: ", state_his)
    print("action_his: ", action_his)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        # 每个状态在图中对应的坐标
        state = state_his[i]
        x = (state % 3) + 0.5
        y = 2.5 - int(state / 3)
        line.set_data(x, y)

    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(state_his), interval=200, repeat=False
    )
    gif_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "maze_test_1.gif"
    )
    anim.save(gif_path, writer="imagemagick")

    print("the end")
