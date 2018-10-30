import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
from nn_model import SimpleNN
from utils import *


# Number of batches
num_batches = 50

# Number of episodes one batch contains
batch_size = 5

input_dim = 6400  # 80x80
hidden_dim = 200
output_dim = 2

# Discount factor
gamma = 0.99

checkpoint_path = os.path.join(os.getcwd(), 'log/simple_nn.ckpt')

save_model_flag = True
save_model_every = 100


if __name__ == "__main__":
    env = gym.make("Pong-v0")

    with tf.Graph().as_default(), tf.Session() as sess:
        policy_network = SimpleNN(input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  output_dim=output_dim,
                                  params={'learning_rate': 1e-3})
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_path))
        if ckpt and ckpt.model_checkpoint_path:
            print("Restoring saved model.")
            saver.restore(sess, checkpoint_path)
        else:
            if not os.path.exists(checkpoint_path):
                os.makedirs(os.path.dirname(checkpoint_path))
            sess.run(tf.global_variables_initializer())
            print("Training new model.")

        observation = env.reset()
        batch_states, batch_actions, batch_rewards = [], [], []
        ep_states, ep_actions, ep_rewards = [], [], []
        prev_ob = None
        episode = 0
        reward_sum = 0.0
        running_reward = None
        episode_results = []  # Used for tracking accuracy.
        i = 0
        while True:
            i += 1
            #print(i, " ", episode)
            curr_ob = preprocess_frame(observation)
            state = curr_ob - prev_ob if prev_ob is not None else np.zeros(input_dim)
            prev_ob = curr_ob
            ep_states.append(state)

            # Forward policy to get next action
            act = policy_network.sample(state, sess)
            #print(act)
            ep_actions.append(act)  # remember action as label
            act = 2 if act == 1 else 3  # convert action label to 'real' action id

            # Step the environment
            observation, reward, done, info = env.step(act)
            reward = float(reward)
            ep_rewards.append(reward)
            reward_sum += reward

            if done:
                episode += 1
                episode_results.append(reward)  # Store result of finished episode.

                discounted_rewards = discount_rewards(ep_rewards, gamma)

                # Normalize discounted rewards
                discounted_rewards -= np.mean(discounted_rewards)
                discounted_rewards /= (np.std(discounted_rewards) + 1e-12)

                # Store current episode into batch
                batch_rewards.extend(discounted_rewards)
                batch_actions.extend(np.array(ep_actions))
                batch_states.extend(np.array(ep_states))

                if episode % batch_size == 0:
                    policy_network.train_step(batch_states, batch_actions, batch_rewards, sess)
                    print("Batch states shape: ")
                    print(len(batch_states))
                    batch_states, batch_actions, batch_rewards = [], [], []
                    print('Episode num: %d reward total was %f. running mean: %f' % (episode, reward_sum, running_reward))

                if save_model_flag and episode % save_model_every == 0:
                    episode_results = np.array(episode_results)
                    accuracy = np.sum(episode_results == 1) / len(episode_results)
                    episode_results = []
                    print("Saving the model. episode num: %d, acc: %f." % (episode, accuracy))
                    saver.save(sess, checkpoint_path)

                ep_states, ep_actions, ep_rewards = [], [], []

                # Tracking reward
                running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

                if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends
                    print(('ep %d: reward: %f reward total: %f. running_reward %f.' % (episode, reward, reward_sum, running_reward)) + \
                          ('' if reward == -1 else '!!!!!!!!!'))
                reward_sum = 0.0
                observation = env.reset()
                prev_ob = None
