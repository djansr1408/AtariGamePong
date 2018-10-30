"""
    This file contains definition of a few kinds of neural networks.
    Author: Srdjan Jovanovic
"""
import tensorflow as tf
import numpy as np


class SimpleNN():
    def __init__(self, input_dim, hidden_dim, output_dim, params):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, input_dim])
        #local1
        with tf.variable_scope('local1') as scope:
            weights = tf.get_variable(name='W1',
                                      shape=[self.input_dim, self.hidden_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.0125, dtype=tf.float32),
                                      dtype=tf.float32)
            # biases = tf.get_variable(name='b1', shape=[hidden_size],
            #                          initializer=tf.constant_initializer(0))
            self.local1 = tf.nn.relu(tf.matmul(self.inputs, weights))

        #local2
        with tf.variable_scope('local2') as scope:
            weights = tf.get_variable(name='W2',
                                      shape=[self.hidden_dim, self.output_dim],
                                      initializer=tf.truncated_normal_initializer(stddev=0.06, dtype=tf.float32),
                                      dtype=tf.float32)
            # biases = tf.get_variable(name='b2', shape=[output_size],
            #                          initializer=tf.constant_initializer(0))
            self.logits = tf.matmul(self.local1, weights)

        self.sample_action = tf.reshape(tf.multinomial(self.logits, 1), [])

        # log probabilities
        log_probs = tf.log(tf.nn.softmax(self.logits))

        # actions in one episode
        self.actions = tf.placeholder(dtype=tf.int32)

        # rewards for each applied action
        self.rewards = tf.placeholder(dtype=tf.float32)

        # get log probabilities for actions in the episode
        indices = tf.range(tf.shape(log_probs)[0]) * tf.shape(log_probs)[1] + self.actions
        log_probs_flattened = tf.reshape(log_probs, [-1])
        action_probs = tf.gather(log_probs_flattened, indices)

        # loss equals log probabilities of actions multiplied with rewards from the actions
        # sign '-' is there for decreasing loss when action is 'good'(increasing loss for 'bad' actions)
        loss = -tf.reduce_sum(tf.multiply(action_probs, self.rewards))

        # Set optimizer
        optimizer = tf.train.RMSPropOptimizer(params['learning_rate'])
        self.train = optimizer.minimize(loss)

    def sample(self, state, sess):
        """
        This function chooses the next action based on current observation.
        It chooses either greedy action from network output or random action with some probability.
        :param state: Current state which is input in the network.
        :param sess: Session for executing graph nodes.
        :return: Action as an integer.
        """
        logits, act = sess.run([self.logits, self.sample_action], feed_dict={self.inputs: [state]})
        return act

    def train_step(self, states, actions, rewards, sess):
        """
        This function runs one train step.
        :param states:
        :param actions:
        :param rewards:
        :param sess:
        :return:
        """
        batch_feed = {self.inputs: states,
                      self.actions: actions,
                      self.rewards: rewards}
        sess.run(self.train, feed_dict=batch_feed)

