# OpenAI Gym CartPole-v1
# SARSA

import gym
import math
from time import sleep
import numpy as np

# observation = [ cart position, cart velocity, pole angle, pole velocity at tip ]
# Из документации: cart position: -2.4 - +2.4
#                  cart velocity: -inf - +inf
#                  pole angle: -41.8 - +41.8
#                  pole velocity at tip: -inf - +inf
# Действия: 0 - движение влево, 1 - движение вправо
# Вознаграждение: 1 за каждый шаг
# Начальное состояние: значения всех параметров взяты из равномерного распределения на [-0.05, 0.05]
# Критерии окончания эпизодв: |pole angle|>12, |cart position|>2.4, длина эпизода>500.

env = gym.make('CartPole-v1')


def q_index(observation, num_buckets):
    # Здесь num_buckets - массив, который показывает как определяются состояния. Так как остояния у нас непрерывные, то
    # соответствующий интервал надо разделить на несколько отрезков.
    # Также в соответствии с данным разделением вычисляется индекс в массиве состояний.

    total_index = np.prod(num_buckets)
    cart_pos, cart_vel, pole_ang, pole_vel = observation
    index = 0

    cart_pos_thr = env.env.x_threshold + 1
    cart_vel_thr = 100
    pole_ang_thr = math.radians(15)
    pole_vel_thr = math.radians(50)

    cart_pos_bins = np.linspace(-cart_pos_thr, cart_pos_thr, num=num_buckets[0] + 1)
    index += (np.digitize(cart_pos, cart_pos_bins) - 1) * total_index / num_buckets[0]

    cart_vel_bins = np.linspace(-cart_vel_thr, cart_vel_thr, num=num_buckets[1] + 1)
    index += (np.digitize(cart_vel, cart_vel_bins) - 1) * total_index / num_buckets[1] / num_buckets[0]

    pole_ang_bins = np.linspace(-pole_ang_thr, pole_ang_thr, num=num_buckets[2] + 1)
    index += (np.digitize(pole_ang, pole_ang_bins) - 1) * total_index / num_buckets[2] / num_buckets[1] / num_buckets[0]

    pole_vel_bins = np.linspace(-pole_vel_thr, pole_vel_thr, num=num_buckets[3] + 1)
    index += (np.digitize(pole_vel, pole_vel_bins) - 1) * total_index / num_buckets[3] / num_buckets[2] / \
        num_buckets[1] / num_buckets[0]

    return np.int(index)


gamma = 0.9             # поправочный коэффициент
epsilon = 1.0           # начальное значение epsilon для применения epsilon-жадной стратегии
epsilon_decay = 0.9999  # множитель значения epsilon для каждого шага
epsilon_min = 0.1       # минимальное значение epsilon
alpha = 0.05            # скорость обучения

num_buckets = np.array([4, 4, 16, 8])  # Разбивка непрерывного пространства на дискретный набор состояний
q_table = np.zeros((np.prod(num_buckets), env.action_space.n))  # Q-массив, инициализируется 0
env._max_episode_steps = 500

episodes = 4000
for i in range(episodes + 1):
    observation = env.reset()
    var_epsilon = epsilon

    for episode in range(500):
        st = q_index(observation, num_buckets)

        # Выбор действия в соответствии с эпсилон-жадной стратегией
        if episode < 20:
            action = env.action_space.sample()
        else:
            if np.random.rand() < var_epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[st])

        observation, reward, done, info = env.step(action)  # Выполняем действие, получаем награду

        if abs(observation[3]) > math.radians(50):
            break

        next_st = q_index(observation, num_buckets)

        # Выбор действия в соответствии с эпсилон-жадной стратегией
        if episode < 20:
            next_action = env.action_space.sample()
        else:
            if np.random.rand() < var_epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(q_table[st])

        var_epsilon = var_epsilon * epsilon_decay
        if var_epsilon < epsilon_min:
            var_epsilon = epsilon_min

        q_table[st, action] = \
            (1 - alpha) * q_table[st, action] + alpha * (reward + gamma * q_table[next_st, next_action])

        if done:
            break

    if i % (episodes / 100) == 0:
        print('{}% completed'.format(i / (episodes / 100)))


score = 0
observation = env.reset()

for _ in range(501):
    env.render()
    st = q_index(observation, num_buckets)
    action = np.argmax(q_table[st])
    observation, reward, done, info = env.step(action)
    score += reward
    sleep(0.1)

    if done:
        break

env.close()
print('Score=', score)
