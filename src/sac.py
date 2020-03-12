import os
import numpy as np
import tensorflow as tf
import math
import tensorflow_probability as tfp
import logging

logging.basicConfig(level=logging.DEBUG)

class SAC:

    def __init__(self, config):
        self.epochs = config['epochs']
        self.learning_rate = config['learning_rate']
        self.target_update = config['target_update']
        self.gamma = config['gamma']
        self.model_name = config['model_name']
        self.seed = config['seed']
        self.log_step = config['log_step']
        self.train_batch_size = config['train_batch_size']
        self.valid_batch_size = config['valid_batch_size']
        self.optimizer = config['optimizer']
        self.initializer = config['initializer']
        self.logs_path = config['logs_path']
        self.SERVING_DIR = os.path.join(self.logs_path, self.model_name+'_serving', '1')
        self.TF_SUMMARY_DIR = os.path.join(self.logs_path, self.model_name+'_summary')
        self.CKPT_DIR = os.path.join(self.logs_path, self.model_name+'_checkpoint')
        self.split = config['split']
        self.action_dim = 1

        self.INITIALIZERS = {
            'xavier': tf.glorot_uniform_initializer(), 
            'uniform': tf.random_uniform_initializer(-1, 1)
        }

        self.OPTIMIZERS = {
            'sgd': tf.train.GradientDescentOptimizer(self.learning_rate),
            'adam': tf.train.AdamOptimizer(self.learning_rate),
            'sgd_mom': tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9, use_nesterov=True),
            'rmsprop': tf.train.RMSPropOptimizer(self.learning_rate),
            'adagrad': tf.train.AdagradOptimizer(self.learning_rate)
        }

        self.LOSSES = {
            'mse': tf.losses.mean_squared_error,
            'huber': tf.losses.huber_loss
        }

        if self.optimizer not in self.OPTIMIZERS.keys():
            raise ValueError("optimizer should be in {}".format(self.OPTIMIZERS.keys()))
        
        if self.logs_path is None:
            raise ValueError("export_dir cannot be empty")

    def input_fn(self, transition_matrices):

        # Fetch current_state, action, reward and next_state matrices.
        current_states, actions, rewards, next_states = transition_matrices

        current_states = current_states.astype(np.float32)
        actions = actions.astype(np.float32)
        rewards = rewards.astype(np.float32)
        next_states = next_states.astype(np.float32)

        # Convert action dtype for indexing.
        actions = actions.astype(np.int32)

        # Split dataset into train and validation set.
        split_percentage = self.split
        num_samples = len(current_states)
        train_size = int(split_percentage * num_samples)
        valid_size = int((1-split_percentage) * num_samples)
        train_set = (current_states[:train_size], actions[:train_size], rewards[:train_size], next_states[:train_size])
        valid_set = (current_states[-valid_size:], actions[-valid_size:], rewards[-valid_size:], next_states[-valid_size:])

        # Calculate number of train batches.
        self.num_train_batches = int(math.ceil(train_size / float(self.train_batch_size)))
        # Calculate number of valid batches.
        self.num_valid_batches = int(math.ceil(valid_size / float(self.valid_batch_size)))

        # Create Dataset object from input.
        train_dataset = tf.data.Dataset.from_tensor_slices(train_set).batch(self.train_batch_size)
        valid_dataset = tf.data.Dataset.from_tensor_slices(valid_set).batch(self.valid_batch_size)

        # Create generic iterator.
        data_iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)

        # Create initialisation operations.
        train_init_op = data_iter.make_initializer(train_dataset)
        valid_init_op = data_iter.make_initializer(valid_dataset)

        return train_init_op, valid_init_op, data_iter

    def value_network(self, current_states, variable_scope):
        """Computes value function at a given state"""
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):

            # Value function estimate for the current state.
            v = tf.layers.dense(current_states, 1, activation=tf.nn.relu)
        return v

    def q_network(self, current_states, actions, variable_scope):
        """Computes the action-value function (Q value) at a given state and
        action"""
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):

            # Concatenate current state and action in a vector and pass it to Q
            # network to observe Q value.
            state_action = tf.concat([current_states, tf.cast(actions,
                                                              dtype=tf.float32)], axis=1)
            q = tf.layers.dense(state_action, 1, activation=tf.nn.relu)

            return q

    def policy_network(self, current_states, variable_scope):
        """Recommends the best action given the current state."""
        with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):

            # Calculate the parameters of a gausssian to select the best action
            # give the current state.
            a = tf.layers.dense(current_states, 10, activation=tf.nn.relu)
            mean = tf.layers.dense(a, self.action_dim)
            std_dev = tf.layers.dense(a, self.action_dim)

            return mean, std_dev

    def log_policy(self, current_states):
        """Computes the log probability of a k dimensional vector""" 
        # Calculate mean and standard deviation of the gaussian.
        mean, std_dev = self.policy_network(current_states,
                                            variable_scope="policy_network")

        # Sample action from the defined gaussian.
        action, log_pi = self.sample_action(mean, std_dev)

        return log_pi, action

    def sample_action(self, mean, std_dev):
        """Samples an action from gaussian."""
        gaussian = tfp.distributions.Normal(loc=mean, scale=std_dev)
        # TODO add tanh squashing here.
        action = gaussian.sample()
        log_prob = gaussian.log_prob(action)
        return action, log_prob

    def soft_value_function_loss(self, current_states):
        """Computes the loss to update soft value function."""
        with tf.name_scope("value_function_loss"):
            v = self.value_network(current_states,
                                   variable_scope="value_network")
            log_pi, actions = self.log_policy(current_states)
            q = tf.stop_gradient(self.q_network(current_states, actions,
                               variable_scope="q_network"))
            soft_v = tf.reduce_sum(q - log_pi)
            v_loss_op = tf.reduce_sum(0.5 * tf.pow((v -
                                                    tf.stop_gradient(soft_v)), 2))
            return v_loss_op

    def soft_q_function_loss(self, current_states, actions, rewards, next_states):
        """Computes the loss to update soft Q function."""
        with tf.name_scope("q_function_loss"):
            v_target = self.value_network(next_states, variable_scope="target_value_network")
            q = self.q_network(current_states, actions,
                               variable_scope="q_network")
            q_target = rewards + self.gamma * tf.reduce_sum(v_target)
            q_loss_op = tf.reduce_sum(0.5 * tf.pow((q -
                                                    tf.stop_gradient(q_target)), 2))
            return q_loss_op

    def policy_network_loss(self, current_states):
        """Computes the KL divergence loss between policy network and Q network"""
        with tf.name_scope("policy_network_loss"):

            log_pi, actions = self.log_policy(current_states)
            q = self.q_network(current_states, actions, variable_scope="q_network")
            policy_loss_op = tf.reduce_sum(log_pi - tf.stop_gradient(q))
            return policy_loss_op

    def optimize_fn(self, v_loss_op, q_loss_op, policy_loss_op):
        """
        Optimization function for the Backpropagation. 
        Dervied class can override this function to implement custom changes to optimization.
        
        Parameters
        ----------
            loss: Tensor shape=[1,1]
                Computed loss for all the samples in batch,
                output of `_loss_fn()`.
        
        Returns
        -------
            optimize_op: Tensorflow Op
                Optimization operation to be performed on loss.
        """
        with tf.variable_scope('optimization'):
            # Select the optimizer.
            optimizer = self.OPTIMIZERS[self.optimizer]

            v_optimize_op = optimizer.minimize(v_loss_op)
            q_optimize_op = optimizer.minimize(q_loss_op)
            policy_optimize_op = optimizer.minimize(policy_loss_op)
            optimize_op = tf.group(v_optimize_op, q_optimize_op, policy_optimize_op)
            return optimize_op


    def train(self, current_states, actions, rewards, next_states):

        # Create loss operation for value function update.
        v_loss_op = self.soft_value_function_loss(current_states)



        # Create loss operation for Q function update.
        q_loss_op = self.soft_q_function_loss(current_states, actions, rewards,
                                             next_states)

        # Create loss operation for policy network update.
        policy_loss_op = self.policy_network_loss(current_states)

        # Combine all the loss operations
        optimize_op = self.optimize_fn(v_loss_op, q_loss_op, policy_loss_op)

        # Log loss in tensorboard summary.
        tf.summary.scalar('loss', v_loss_op)
        tf.summary.scalar('loss', q_loss_op)
        tf.summary.scalar('loss', policy_loss_op)
        
        # Summaries for all the trainable variables.
        #utils.parameter_summaries(tf.trainable_variables())

        # TODO: Add tensorboard model evaluation metrics.

        summary = tf.summary.merge_all()

        return optimize_op, v_loss_op, q_loss_op, policy_loss_op, summary

    def copy(self, primary_scope, target_scope):

        with tf.name_scope("copy"):

            primary_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=primary_scope)
            target_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=target_scope)

            primary_variables_sorted = sorted(primary_variables, key=lambda v: v.name)
            target_variables_sorted = sorted(target_variables, key=lambda v: v.name)

            assign_ops = []

            for primary_var, target_var in zip(primary_variables_sorted, target_variables_sorted):
                assign_ops.append(target_var.assign(tf.identity(primary_var)))

            copy_op = tf.group(*assign_ops)

            return copy_op
