"""
    This file contains code for simulating game using trained model.
    It restores model and gets next actions from it.
"""
import tensorflow as tf
import numpy as np
import gym
import os
import time
from nn_model import SimpleNN
from utils import *

# Directory where checkpoint file is stored.
checkpoint_dir = 'log'

# Model parameters used for training.
input_dim = 6400  # 80x80
hidden_dim = 200
output_dim = 2
params = {'learning_rate': 1e-3}


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), checkpoint_dir)
    print(path)
    checkpoint_file = tf.train.latest_checkpoint(path)

    tf.reset_default_graph()
    sess = tf.Session()

    env = gym.make("Pong-v0")

    agent = SimpleNN(input_dim=input_dim,
                     hidden_dim=hidden_dim,
                     output_dim=output_dim,
                     params=params)

    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_file)

    prev_ob = None
    observation = env.reset()
    while True:
        env.render()
        time.sleep(0.01)
        # Get state
        curr_ob = preprocess_frame(observation)
        state = curr_ob - prev_ob if prev_ob is not None else np.zeros(input_dim)
        prev_ob = curr_ob

        # Get next action
        action = agent.sample(state, sess)  # Returns action label, not the true ID.
        action = 2 if action == 1 else 3    # Convert label to action ID.

        observation, reward, done, info = env.step(action)

        if done:
            time.sleep(5)
            observation = env.reset()
            if reward == 1:
                print("Won !!!")
            else:
                print("Lost !!!")
