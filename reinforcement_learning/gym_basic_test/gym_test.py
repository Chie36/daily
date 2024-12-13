import gym
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import os


def display_frames_to_video(frames):
    print("frames size:{}".format(len(frames)))
    plt.figure(figsize=(frames[0].shape[0] / 72, frames[0].shape[1] / 72), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate_func(frame):
        patch.set_data(frame)

    anim = animation.FuncAnimation(
        plt.gcf(), animate_func, frames=frames[1:], interval=50
    )
    gif_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gym_test.gif")
    anim.save(gif_path, writer="imagemagick")


def gym_test(is_save=True, epoch_num=3):
    env = (
        gym.make("CartPole-v1", render_mode="rgb_array")
        if is_save
        else gym.make("CartPole-v1", render_mode="human")
    )
    frames = []
    for epoch in range(1, 1 + epoch_num):
        score = 0
        state = env.reset()
        for i in range(1000):
            frames.append(env.render())
            action = env.action_space.sample()
            # print("input state:{}, action:{}".format(state, action))
            state, reward, terminated, truncated, info = env.step(action)
            # print(
            #     "output state:{}, reward:{}, terminated:{}, truncated:{}, info:{}".format(
            #         state, reward, terminated, truncated, info
            #     )
            # )
            score += reward
            time.sleep(0.5)
            if terminated or truncated:
                print("finished after {} steps, final score:{}".format(i, score))
                break
    if is_save:
        display_frames_to_video(frames)
    env.close()


if __name__ == "__main__":
    gym_test()
    print("the end")
