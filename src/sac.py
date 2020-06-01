import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np

tf.keras.backend.set_floatx('float64')

EPSILON = 1e-16

class Actor(Model):

    def __init__(self, action_dim):
        super().__init__()
        self.action_dim = action_dim
        self.dense1_layer = layers.Dense(32, activation=tf.nn.relu)
        self.dense2_layer = layers.Dense(32, activation=tf.nn.relu)
        self.mean_layer = layers.Dense(self.action_dim)
        self.stdev_layer = layers.Dense(self.action_dim)

    def call(self, state):
        # Get mean and standard deviation from the policy network
        a1 = self.dense1_layer(state)
        a2 = self.dense2_layer(a1)
        mu = self.mean_layer(a2)

        # Standard deviation is bounded by a constraint of being non-negative
        # therefore we produce log stdev as output which can be [-inf, inf]
        log_sigma = self.stdev_layer(a2)
        sigma = tf.exp(log_sigma)

        # Use re-parameterization trick to deterministically sample action from
        # the policy network. First, sample from a Normal distribution of
        # sample size as the action and multiply it with stdev
        dist = tfp.distributions.Normal(mu, sigma)
        action_ = dist.sample()

        # Apply the tanh squashing to keep the gaussian bounded in (-1,1)
        action = tf.tanh(action_)

        # Calculate the log probability
        log_pi_ = dist.log_prob(action_)
        # Change log probability to account for tanh squashing as mentioned in
        # Appendix C of the paper
        log_pi = log_pi_ - tf.reduce_sum(tf.math.log(1 - action**2 + EPSILON), axis=1,
                                         keepdims=True)

        return action, log_pi

    @property
    def trainable_variables(self):
        return self.dense1_layer.trainable_variables + \
                self.dense1_layer.trainable_variables + \
                self.mean_layer.trainable_variables + \
                self.stdev_layer.trainable_variables

class Critic(Model):

    def __init__(self):
        super().__init__()
        self.dense1_layer = layers.Dense(32, activation=tf.nn.relu)
        self.dense2_layer = layers.Dense(32, activation=tf.nn.relu)
        self.output_layer = layers.Dense(1)

    def call(self, state, action):
        state_action = tf.concat([state, action], axis=1)
        a1 = self.dense1_layer(state_action)
        a2 = self.dense2_layer(a1)
        q = self.output_layer(a2)
        return q

    @property
    def trainable_variables(self):
        return self.dense1_layer.trainable_variables + \
                self.output_layer.trainable_variables + \
                self.dense2_layer.trainable_variables


class SoftActorCritic:

    def __init__(self, action_dim, writer, epoch_step=1, learning_rate=0.0003,
                 alpha=0.2, gamma=0.99,
                polyak=0.995):
        self.policy = Actor(action_dim)
        self.q1 = Critic()
        self.q2 = Critic()
        self.target_q1 = Critic()
        self.target_q2 = Critic()

        self.writer = writer
        self.epoch_step = epoch_step

        self.alpha = tf.Variable(0.0, dtype=tf.float64)
        self.target_entropy = -tf.constant(action_dim, dtype=tf.float64)
        self.gamma = gamma
        self.polyak = polyak

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate)


    def sample_action(self, current_state):
        current_state_ = np.array(current_state, ndmin=2)
        action, _ = self.policy(current_state_)
        return action[0]


    def update_q_network(self, current_states, actions, rewards, next_states, ends):

        with tf.GradientTape() as tape1:
            # Get Q value estimates, action used here is from the replay buffer
            q1 = self.q1(current_states, actions)

            # Sample actions from the policy for next states
            pi_a, log_pi_a = self.policy(next_states)

            # Get Q value estimates from target Q network
            q1_target = self.target_q1(next_states, pi_a)
            q2_target = self.target_q2(next_states, pi_a)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - self.alpha * log_pi_a
            y = tf.stop_gradient(rewards + self.gamma * ends * soft_q_target)

            critic1_loss = tf.reduce_mean((q1 - y)**2)

        with tf.GradientTape() as tape2:
            # Get Q value estimates, action used here is from the replay buffer
            q2 = self.q2(current_states, actions)

            # Sample actions from the policy for next states
            pi_a, log_pi_a = self.policy(next_states)

            # Get Q value estimates from target Q network
            q1_target = self.target_q1(next_states, pi_a)
            q2_target = self.target_q2(next_states, pi_a)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q_target = tf.minimum(q1_target, q2_target)

            # Add the entropy term to get soft Q target
            soft_q_target = min_q_target - self.alpha * log_pi_a
            y = tf.stop_gradient(rewards + self.gamma * ends * soft_q_target)

            critic2_loss = tf.reduce_mean((q2 - y)**2)

        grads1 = tape1.gradient(critic1_loss, self.q1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(grads1,
                                                   self.q1.trainable_variables))

        grads2 = tape2.gradient(critic2_loss, self.q2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(grads2,
                                                   self.q2.trainable_variables))

        with self.writer.as_default():
            for grad, var in zip(grads1, self.q1.trainable_variables):
                tf.summary.histogram(f"grad-{var.name}", grad, self.epoch_step)
                tf.summary.histogram(f"var-{var.name}", var, self.epoch_step)
            for grad, var in zip(grads2, self.q2.trainable_variables):
                tf.summary.histogram(f"grad-{var.name}", grad, self.epoch_step)
                tf.summary.histogram(f"var-{var.name}", var, self.epoch_step)

        return critic1_loss, critic2_loss

    def update_policy_network(self, current_states):
        with tf.GradientTape() as tape:
            # Sample actions from the policy for current states
            pi_a, log_pi_a = self.policy(current_states)

            # Get Q value estimates from target Q network
            q1 = self.q1(current_states, pi_a)
            q2 = self.q2(current_states, pi_a)

            # Apply the clipped double Q trick
            # Get the minimum Q value of the 2 target networks
            min_q = tf.minimum(q1, q2)

            soft_q = min_q - self.alpha * log_pi_a

            actor_loss = -tf.reduce_mean(soft_q)

        variables = self.policy.trainable_variables
        grads = tape.gradient(actor_loss, variables)
        self.actor_optimizer.apply_gradients(zip(grads, variables))

        with self.writer.as_default():
            for grad, var in zip(grads, variables):
                tf.summary.histogram(f"grad-{var.name}", grad, self.epoch_step)
                tf.summary.histogram(f"var-{var.name}", var, self.epoch_step)

        return actor_loss


    def update_alpha(self, current_states):
        with tf.GradientTape() as tape:
            # Sample actions from the policy for current states
            pi_a, log_pi_a = self.policy(current_states)

            alpha_loss = tf.reduce_mean( - self.alpha*(log_pi_a +
                                                       self.target_entropy))

        variables = [self.alpha]
        grads = tape.gradient(alpha_loss, variables)
        self.alpha_optimizer.apply_gradients(zip(grads, variables))

        with self.writer.as_default():
            for grad, var in zip(grads, variables):
                tf.summary.histogram(f"grad-{var.name}", grad, self.epoch_step)
                tf.summary.histogram(f"var-{var.name}", var, self.epoch_step)

        return alpha_loss


    def train(self, current_states, actions, rewards, next_states, ends):

        # Update Q network weights
        critic1_loss, critic2_loss = self.update_q_network(current_states, actions, rewards, next_states, ends)

        # Update policy network weights
        actor_loss = self.update_policy_network(current_states)

        alpha_loss = self.update_alpha(current_states)

        # Update target Q network weights
        #self.update_weights()

        #if self.epoch_step % 10 == 0:
        #    self.alpha = max(0.1, 0.9**(1+self.epoch_step/10000))
        #    print("alpha: ", self.alpha, 1+self.epoch_step/10000)

        return critic1_loss, critic2_loss, actor_loss, alpha_loss

    def update_weights(self):

        for theta_target, theta in zip(self.target_q1.trainable_variables,
                                       self.q1.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta

        for theta_target, theta in zip(self.target_q2.trainable_variables,
                                       self.q2.trainable_variables):
            theta_target = self.polyak * theta_target + (1 - self.polyak) * theta
