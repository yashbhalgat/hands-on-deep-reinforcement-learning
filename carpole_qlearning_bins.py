import gym
import numpy as np
import matplotlib.pyplot as plt


def hash_state(quantized_s):
    return int("".join([str(int(s)) for s in quantized_s]))

class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4, 2.4, 9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity_bins = np.linspace(-3.5, 3.5, 9)

    def transform(self, state):
        cart_pos, cart_vel, pole_angle, pole_vel = state
        return hash_state([
            np.digitize(x=[cart_pos], bins=self.cart_position_bins)[0],
            np.digitize(x=[cart_vel], bins=self.cart_velocity_bins)[0],
            np.digitize(x=[pole_angle], bins=self.pole_angle_bins)[0],
            np.digitize(x=[pole_vel], bins=self.pole_velocity_bins)
        ])

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        num_actions = env.action_space.n
        num_states = 10**env.observation_space.shape[0]
        self.Q = np.random.uniform(low=-1, high=-1, size=(num_states, num_actions))
        self.alpha = 0.005

    def predict(self, s):
        hashed_s = self.feature_transformer.transform(s)
        return self.Q[hashed_s]     #returns array

    def update(self, s, a, G):
        hashed_s = self.feature_transformer.transform(s)
        self.Q[hashed_s, a] += self.alpha*(G-self.Q[hashed_s, a])

    def sample_action(self, s, eps):
        if np.random.random()<eps:
            return self.env.action_space.sample()
        else:
            p = self.predict(s)
            return np.argmax(p)


def play_one_episode(model, eps, gamma):
    observation = model.env.reset()
    done = False
    total_reward = 0
    iters = 0

    while not done and iters<10000:
        action = model.sample_action(observation, eps)
        next_observation, reward, done, _ = model.env.step(action)

        total_reward += reward

        if done and iters<199:
            reward = -300

        # calculate update
        G = reward + gamma*np.max(model.Q[model.feature_transformer.transform(next_observation)])
        model.update(observation, action, G)
        observation = next_observation

        iters += 1

    return total_reward

env = gym.make("CartPole-v0")
feature_transformer = FeatureTransformer()
model = Model(env, feature_transformer)
gamma = 0.9

total_rewards = np.empty(10000)
for i in range(10000):
    eps = 1./(i+1)**0.5
    total_rewards[i] = play_one_episode(model, eps, gamma)
    if i%100==0:
        print(total_rewards[i], eps)

plt.plot(total_rewards)
plt.show()
