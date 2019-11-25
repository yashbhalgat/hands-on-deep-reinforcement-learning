import gym
import numpy as np
import matplotlib.pyplot as plt


def get_action(s, w):
    return 1 if s.dot(w)>0 else 0

def play_one_episode(env, params):
    observation = env.reset()
    done = False
    t = 0

    while not done and t<1000:
        t += 1
        action = get_action(observation, params)
        observation, reward, done, _ = env.step(action)
        if done:
            break

    return t

def play_mult_episodes(env, T, params):
    episode_lengths = np.empty(T)
    for i in range(T):
        episode_lengths[i] = play_one_episode(env, params)

    avg_length = episode_lengths.mean()
    print("average length:", avg_length)
    return avg_length

def random_search(env):
    best = 0
    best_params = None
    lengths = []
    for t in range(100):
        params = np.random.random(4)*2-1
        avg_length = play_mult_episodes(env, 100, params)
        lengths.append(avg_length)

        if avg_length>best:
            best_params = params
            best = avg_length

    return lengths, best_params

env = gym.make("CartPole-v0")
lengths, best_params = random_search(env)
plt.plot(lengths)
plt.show()

play_mult_episodes(env, 100, best_params)
