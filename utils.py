import numpy as np

def preprocess_frame(I):
    """
    Preprocess input image(frame) of shape 210x160x3 into 6400(80x80) 1D float vector.
    :param I: input frame of shape 210x160x3 uint8
    :return: 1D flaot vector of length 6400
    """
    I = I[35:195]
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (type 1)
    I[I == 109] = 0  # erase background (type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


# def discount_rewards(rewards, gamma):
#     """
#     Takes 1D array/list of rewards and computes discounted reward.
#     :param r: 1D array of rewards
#     :param gamma: Discount factor
#     :return: Discounted reward array/list
#     """
#     cumulative_reward = 0
#     dis_rewards = np.zeros(len(rewards), dtype=np.float32)
#     for t in reversed(range(len(rewards))):
#         if rewards[t] != 0:  # this is just for Pong game!!!
#             cumulative_reward = 0
#         cumulative_reward = gamma * cumulative_reward + rewards[t]
#         dis_rewards[t] = cumulative_reward
#     return dis_rewards

def discount_rewards(r, gamma):
    """
    Takes 1D array of rewards and computes discounted reward.
    :param r: 1D array of rewards
    :return: discounted reward
    """
    r = np.array(r)
    discount_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0  # reset the sum, since this was a game boundary(Pong specific!)
        running_add = running_add * gamma + r[t]
        discount_r[t] = running_add
    return discount_r